def reconstruct_trajectory_from_imu(accel_data, gyro_data, dt):
    """
    从IMU数据重建轨迹
    accel_data: N×3数组，加速度数据 (m/s²)
    gyro_data: N×3数组，角速度数据 (rad/s)
    dt: 采样时间间隔
    """
    num_samples = len(accel_data)
    
    # 初始化变量
    position = np.zeros((num_samples, 3))
    velocity = np.zeros((num_samples, 3))
    orientation = np.zeros((num_samples, 3))  # 欧拉角 [phi, theta, psi]
    
    # 初始方向假设为 identity
    R = np.eye(3)
    
    # 重力矢量
    gravity = np.array([0, 0, 9.81])
    
    for i in range(1, num_samples):
        # 当前角速度
        omega = gyro_data[i]
        
        # 更新方向 (简化旋转积分)
        # 使用角速度更新旋转矩阵
        omega_skew = np.array([[0, -omega[2], omega[1]],
                              [omega[2], 0, -omega[0]],
                              [-omega[1], omega[0], 0]])
        
        R = R @ (np.eye(3) + omega_skew * dt)
        
        # 正交化旋转矩阵
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt
        
        # 将加速度转换到全局坐标系，并减去重力
        global_accel = R @ accel_data[i] - gravity
        
        # 积分得到速度
        velocity[i] = velocity[i-1] + global_accel * dt
        
        # 积分得到位置
        position[i] = position[i-1] + velocity[i] * dt
        
        # 从旋转矩阵提取欧拉角 (可选)
        # orientation[i] = rotation_matrix_to_euler_angles(R)
    
    return position, velocity, orientation

# 比较函数
def compare_trajectories(simulated_pos, measured_pos):
    """
    比较模拟轨迹和实测轨迹
    """
    # 确保两个轨迹有相同的时间点
    min_len = min(len(simulated_pos), len(measured_pos))
    simulated_pos = simulated_pos[:min_len]
    measured_pos = measured_pos[:min_len]
    
    # 计算误差指标
    errors = np.linalg.norm(simulated_pos - measured_pos, axis=1)
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    
    print(f"平均位置误差: {mean_error:.3f} m")
    print(f"最大位置误差: {max_error:.3f} m")
    
    # 绘制比较图
    fig = plt.figure(figsize=(15, 10))
    
    # 3D轨迹比较
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.plot(simulated_pos[:, 0], simulated_pos[:, 1], simulated_pos[:, 2], 'b-', label='模拟轨迹')
    ax1.plot(measured_pos[:, 0], measured_pos[:, 1], measured_pos[:, 2], 'r-', label='实测轨迹')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D轨迹比较')
    ax1.legend()
    
    # 各轴位置比较
    time = np.arange(min_len) * dt
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(time, simulated_pos[:, 0], 'b-', label='模拟 X')
    ax2.plot(time, measured_pos[:, 0], 'b--', label='实测 X')
    ax2.plot(time, simulated_pos[:, 1], 'r-', label='模拟 Y')
    ax2.plot(time, measured_pos[:, 1], 'r--', label='实测 Y')
    ax2.plot(time, simulated_pos[:, 2], 'g-', label='模拟 Z')
    ax2.plot(time, measured_pos[:, 2], 'g--', label='实测 Z')
    ax2.set_xlabel('时间 (s)')
    ax2.set_ylabel('位置 (m)')
    ax2.set_title('各轴位置比较')
    ax2.legend()
    
    # 误差随时间变化
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(time, errors, 'k-')
    ax3.set_xlabel('时间 (s)')
    ax3.set_ylabel('误差 (m)')
    ax3.set_title('位置误差随时间变化')
    ax3.grid(True)
    
    # 误差分布直方图
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.hist(errors, bins=50, alpha=0.7)
    ax4.set_xlabel('误差 (m)')
    ax4.set_ylabel('频率')
    ax4.set_title('误差分布')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return errors