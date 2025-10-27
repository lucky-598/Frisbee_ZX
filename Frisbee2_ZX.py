import serial
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
from scipy.spatial.transform import Rotation as R

class FrisbeeTrajectory:
    def __init__(self, com_port='COM5', baudrate=9600):
        # 初始化蓝牙串口连接
        try:
            self.ser = serial.Serial(com_port, baudrate, timeout=1)
            time.sleep(2)  # 等待连接稳定
            print(f"Connected to {com_port}")
        except serial.SerialException as e:
            print(f"Error opening serial port: {e}")
            self.ser = None

        # 数据存储列表
        self.timestamps = []
        self.accel_data = []  # 存储原始加速度数据 (ax, ay, az)
        self.gyro_data = []   # 存储原始角速度数据 (gx, gy, gz)
        
        # 轨迹状态变量
        self.velocity = np.zeros(3)  # 速度向量 [vx, vy, vz]
        self.position = np.zeros(3)  # 位置向量 [x, y, z]
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # 四元数表示姿态

        # 传感器参数（需根据你的MPU6050配置和校准结果调整）
        self.accel_scale = 16384.0  # ±2g量程下的灵敏度 LSB/g
        self.gyro_scale = 131.0     # ±250deg/s量程下的灵敏度 LSB/(deg/s)
        self.dt = 0.01             # 采样时间间隔 (s)，应与Arduino发送速率匹配

        # 物理常数
        self.g = np.array([0, 0, -9.81])  # 重力加速度向量 (m/s^2)
        
        # 飞盘空气动力学参数（简化模型，需要根据你的飞盘查阅文献或实验确定）
        self.drag_coeff = 0.08     # 阻力系数
        self.lift_coeff = 0.15     # 升力系数

    def read_serial_data(self):
        """从蓝牙串口读取并解析一帧数据"""
        if self.ser and self.ser.in_waiting > 0:
            try:
                line = self.ser.readline().decode('utf-8').strip()
                data = line.split(',')
                if len(data) == 6:
                    # 解析原始数据
                    ax_raw = int(data[0]); ay_raw = int(data[1]); az_raw = int(data[2])
                    gx_raw = int(data[3]); gy_raw = int(data[4]); gz_raw = int(data[5])
                    
                    # 转换为物理单位
                    accel = np.array([ax_raw, ay_raw, az_raw]) / self.accel_scale # 转换为 g
                    accel = accel * 9.81  # 转换为 m/s^2
                    
                    gyro = np.array([gx_raw, gy_raw, gz_raw]) / self.gyro_scale # 转换为 deg/s
                    gyro = np.radians(gyro)  # 转换为 rad/s
                    
                    current_time = time.time()
                    
                    # 存储数据
                    if len(self.timestamps) > 0:
                        self.dt = current_time - self.timestamps[-1] # 计算实际时间间隔
                    self.timestamps.append(current_time)
                    self.accel_data.append(accel)
                    self.gyro_data.append(gyro)
                    
                    return True
            except (ValueError, UnicodeDecodeError) as e:
                print(f"Data parsing error: {e}")
        return False

    def update_orientation(self, gyro):
        """使用四元数更新姿态（基于陀螺仪数据）"""
        # 四元数更新算法
        q = self.quaternion
        w = gyro  # 当前角速度

        # 四元数微分方程
        q_dot = 0.5 * np.array([
            -q[1]*w[0] - q[2]*w[1] - q[3]*w[2],
             q[0]*w[0] + q[2]*w[2] - q[3]*w[1],
             q[0]*w[1] - q[1]*w[2] + q[3]*w[0],
             q[0]*w[2] + q[1]*w[1] - q[2]*w[0]
        ])
        
        # 一阶积分更新四元数
        new_q = q + q_dot * self.dt
        self.quaternion = new_q / np.linalg.norm(new_q)  # 归一化

    def rotate_vector(self, vector):
        """使用当前四元数旋转一个向量（从机体坐标系到世界坐标系）"""
        q = self.quaternion
        # 将四元数转换为旋转矩阵
        rot_matrix = np.array([
            [1-2*(q[2]**2+q[3]**2), 2*(q[1]*q[2]-q[0]*q[3]), 2*(q[1]*q[3]+q[0]*q[2])],
            [2*(q[1]*q[2]+q[0]*q[3]), 1-2*(q[1]**2+q[3]**2), 2*(q[2]*q[3]-q[0]*q[1])],
            [2*(q[1]*q[3]-q[0]*q[2]), 2*(q[2]*q[3]+q[0]*q[1]), 1-2*(q[1]**2+q[2]**2)]
        ])
        return np.dot(rot_matrix, vector)

    def calculate_dynamic_forces(self, velocity_world):
        """计算空气动力（阻力和升力）"""
        # 阻力：与速度平方成正比，方向相反
        speed = np.linalg.norm(velocity_world)
        if speed > 0:
            drag_direction = -velocity_world / speed
            drag_force = 0.5 * 1.225 * self.drag_coeff * speed**2 * drag_direction # 1.225为空气密度(kg/m^3)
        else:
            drag_force = np.zeros(3)
        
        # 升力：垂直于速度方向和飞盘旋转轴（简化模型，实际更复杂）
        # 这是一个非常简化的模型，实际的飞盘升力与攻角、旋转速度等有关
        lift_force = np.zeros(3)
        # ... (升力计算需要更复杂的空气动力学模型)

        return drag_force #, lift_force

    def update_trajectory(self):
        """更新飞盘轨迹（位置和速度）"""
        if len(self.accel_data) < 2:
            return

        # 获取最新的传感器数据
        current_accel = self.accel_data[-1]
        current_gyro = self.gyro_data[-1]

        # 1. 更新姿态（机体坐标系 -> 世界坐标系）
        self.update_orientation(current_gyro)
        
        # 2. 将加速度矢量从机体坐标系旋转到世界坐标系
        accel_world = self.rotate_vector(current_accel)
        
        # 3. 减去重力加速度（在世界坐标系中是常数）
        # 假设加速度计测量的是机体坐标系的“比力”，包含重力和运动加速度
        linear_accel = accel_world - self.g
        
        # 4. 计算空气动力（需要速度信息）
        # drag_force = self.calculate_dynamic_forces(self.velocity)
        # 将空气动力产生的加速度加到线性加速度上 (F=ma -> a=F/m)
        # 假设飞盘质量m=0.15kg
        # linear_accel = linear_accel + drag_force / 0.15

        # 5. 积分得到速度和位置
        self.velocity += linear_accel * self.dt
        self.position += self.velocity * self.dt

    def run_simulation(self, duration=10):
        """运行主仿真循环"""
        start_time = time.time()
        print("Starting simulation...")
        
        while (time.time() - start_time) < duration:
            if self.read_serial_data():
                self.update_trajectory()
            # 可以添加实时绘图代码 here (使用matplotlib动画)
        
        print("Simulation completed.")
        self.ser.close()

    def plot_results(self):
        """绘制轨迹结果"""
        # 将数据转换为numpy数组便于处理
        positions = np.array([self.position]) # 这里需要根据你的数据存储方式调整
        # ... 绘图代码

# 使用示例
if __name__ == "__main__":
    # 需要根据你的电脑修改COM端口
    # 在Windows设备管理器中查看端口；Linux/Mac通常是 /dev/tty.HC-05-DevB 之类的形式
    trajectory_sim = FrisbeeTrajectory(com_port='COM5') 
    trajectory_sim.run_simulation(duration=10) # 运行10秒
    trajectory_sim.plot_results()