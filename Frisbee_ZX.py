import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp

class FrisbeeTrajectory:
    def __init__(self):
        # 物理常数
        self.g = 9.81  # 重力加速度 (m/s^2)
        self.rho = 1.225  # 空气密度 (kg/m^3)
        
        # 飞盘参数 (标准飞盘尺寸)
        self.m = 0.175  # 质量 (kg)
        self.diameter = 0.27  # 直径 (m)
        self.area = np.pi * (self.diameter/2)**2  # 截面积 (m^2)
        
        # 空气动力学系数
        self.Cd0 = 0.18  # 零升阻力系数
        self.Cl0 = 0.33  # 零攻角升力系数
        self.Cm0 = 0.0   # 零攻角力矩系数
        
        # 初始化状态变量 [x, y, z, vx, vy, vz, phi, theta, psi, omega_x, omega_y, omega_z]
        self.initial_state = np.zeros(12)
        
    def aerodynamic_forces(self, v, alpha, phi):
        """
        计算空气动力和力矩
        v: 速度大小
        alpha: 攻角 (弧度)
        phi: 滚转角 (弧度)
        """
        # 动态压力
        q = 0.5 * self.rho * v**2 * self.area
        
        # 升力系数 (随攻角变化)
        Cl = self.Cl0 * np.cos(alpha)**2
        
        # 阻力系数 (随攻角变化)
        Cd = self.Cd0 * (1 + 4.0 * alpha**2)
        
        # 升力 (垂直于速度方向)
        L = q * Cl
        
        # 阻力 (平行于速度方向，相反)
        D = q * Cd
        
        # 力矩 (简化模型)
        M = q * self.diameter * self.Cm0 * alpha
        
        return L, D, M
    
    def equations_of_motion(self, t, state):
        """
        定义运动微分方程
        state: [x, y, z, vx, vy, vz, phi, theta, psi, omega_x, omega_y, omega_z]
        """
        # 解包状态变量
        x, y, z, vx, vy, vz, phi, theta, psi, omega_x, omega_y, omega_z = state
        
        # 速度大小
        v = np.sqrt(vx**2 + vy**2 + vz**2)
        
        if v < 1e-6:  # 避免除以零
            return np.zeros(12)
        
        # 计算攻角 (速度矢量与飞盘平面夹角)
        velocity_dir = np.array([vx, vy, vz]) / v
        
        # 飞盘法向量 (基于欧拉角)
        R = self.rotation_matrix(phi, theta, psi)
        normal = R[:, 2]  # z轴在全局坐标系中的方向
        
        # 攻角计算
        alpha = np.arccos(np.clip(np.dot(velocity_dir, normal), -1.0, 1.0))
        
        # 空气动力
        L, D, M = self.aerodynamic_forces(v, alpha, phi)
        
        # 阻力方向 (与速度相反)
        drag_dir = -velocity_dir
        
        # 升力方向 (垂直于速度方向和法向量的叉积)
        lift_dir = np.cross(velocity_dir, np.cross(normal, velocity_dir))
        if np.linalg.norm(lift_dir) > 1e-6:
            lift_dir = lift_dir / np.linalg.norm(lift_dir)
        
        # 合力
        F_gravity = np.array([0, 0, -self.m * self.g])
        F_drag = D * drag_dir
        F_lift = L * lift_dir
        F_total = F_gravity + F_drag + F_lift
        
        # 角加速度 (简化模型)
        I = np.array([0.001, 0.001, 0.002])  # 转动惯量 (简化估计)
        tau = np.array([0, M, 0])  # 主要考虑俯仰力矩
        
        # 状态导数
        dxdt = np.zeros(12)
        dxdt[0:3] = [vx, vy, vz]  # 位置变化率
        dxdt[3:6] = F_total / self.m  # 速度变化率
        dxdt[6:9] = self.euler_rates(phi, theta, omega_x, omega_y, omega_z)  # 欧拉角变化率
        dxdt[9:12] = tau / I  # 角速度变化率
        
        return dxdt
    
    def rotation_matrix(self, phi, theta, psi):
        """
        计算旋转矩阵 (ZYX欧拉角)
        """
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(phi), -np.sin(phi)],
                        [0, np.sin(phi), np.cos(phi)]])
        
        R_y = np.array([[np.cos(theta), 0, np.sin(theta)],
                        [0, 1, 0],
                        [-np.sin(theta), 0, np.cos(theta)]])
        
        R_z = np.array([[np.cos(psi), -np.sin(psi), 0],
                        [np.sin(psi), np.cos(psi), 0],
                        [0, 0, 1]])
        
        return R_z @ R_y @ R_x
    
    def euler_rates(self, phi, theta, omega_x, omega_y, omega_z):
        """
        计算欧拉角变化率
        """
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        cos_theta = np.cos(theta)
        tan_theta = np.tan(theta)
        
        dphi = omega_x + sin_phi * tan_theta * omega_y + cos_phi * tan_theta * omega_z
        dtheta = cos_phi * omega_y - sin_phi * omega_z
        dpsi = (sin_phi / cos_theta) * omega_y + (cos_phi / cos_theta) * omega_z
        
        return np.array([dphi, dtheta, dpsi])
    
    def simulate(self, initial_conditions, t_span, t_eval):
        """
        模拟飞盘轨迹
        initial_conditions: 初始条件 [x0, y0, z0, vx0, vy0, vz0, phi0, theta0, psi0, omega_x0, omega_y0, omega_z0]
        t_span: 时间范围 [t_start, t_end]
        t_eval: 评估时间点
        """
        self.initial_state = initial_conditions
        
        # 解微分方程
        sol = solve_ivp(self.equations_of_motion, t_span, self.initial_state, 
                        t_eval=t_eval, method='RK45', rtol=1e-6, atol=1e-9)
        
        return sol
    
    def plot_trajectory(self, sol):
        """
        绘制3D轨迹
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.plot(sol.y[0], sol.y[1], sol.y[2], 'b-', label='模拟轨迹')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('飞盘飞行轨迹')
        ax.legend()
        
        plt.show()

# 使用示例
if __name__ == "__main__":
    # 创建飞盘轨迹模拟器
    simulator = FrisbeeTrajectory()
    
    # 设置初始条件 (根据实际投掷调整)
    # [x0, y0, z0, vx0, vy0, vz0, phi0, theta0, psi0, omega_x0, omega_y0, omega_z0]
    initial_conditions = [0, 0, 1.5, 10, 0, 5, 0, 0.2, 0, 0, 10, 0]
    
    # 设置时间范围
    t_span = [0, 5]  # 5秒模拟
    t_eval = np.linspace(0, 5, 500)  # 500个时间点
    
    # 运行模拟
    sol = simulator.simulate(initial_conditions, t_span, t_eval)
    
    # 绘制轨迹
    simulator.plot_trajectory(sol)