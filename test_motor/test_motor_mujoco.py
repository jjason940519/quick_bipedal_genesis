import sys
import os
import torch
import mujoco
import mujoco.viewer
import time
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt

# 加载 mujoco 模型
m = mujoco.MjModel.from_xml_path('quick_scence.xml')
d = mujoco.MjData(m)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def world2self(quat, v):
    q_w = quat[0] 
    q_vec = quat[1:] 
    v_vec = torch.tensor(v, device=device,dtype=torch.float32)
    a = v_vec * (2.0 * q_w**2 - 1.0)
    b = torch.linalg.cross(q_vec, v_vec) * q_w * 2.0
    c = q_vec * torch.dot(q_vec, v_vec) * 2.0
    result = a - b + c
    return result.to(device)

def get_sensor_data(sensor_name):
    sensor_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
    if sensor_id == -1:
        raise ValueError(f"Sensor '{sensor_name}' not found in model!")
    start_idx = m.sensor_adr[sensor_id]
    dim = m.sensor_dim[sensor_id]
    sensor_values = d.sensordata[start_idx : start_idx + dim]
    return torch.tensor(
        sensor_values, 
        device=device, 
        dtype=torch.float32
    )

def run_thigh(runing_time, process_time, test_radian_max, test_radian_min):
    if runing_time < process_time:
        runing_time += m.opt.timestep * 5
        phase = np.tanh(runing_time / 3)
        d.ctrl[2] = phase * test_radian_max + (1 - phase) * test_radian_min
    elif runing_time < process_time * 2:
        runing_time += m.opt.timestep * 5
        phase = np.tanh((runing_time - process_time) / 3)
        d.ctrl[2] = phase * test_radian_min + (1 - phase) * test_radian_max

    return runing_time

def run_calf(runing_time, process_time, test_radian_max, test_radian_min):
    if runing_time < process_time:
        runing_time += m.opt.timestep * 5
        phase = np.tanh(runing_time / 3)
        d.ctrl[3] = phase * test_radian_max + (1 - phase) * test_radian_min
    else:
        runing_time += m.opt.timestep * 5
        phase = np.tanh((runing_time - process_time) / 3)
        d.ctrl[3] = phase * test_radian_min + (1 - phase) * test_radian_max

    return runing_time

def test_wheel(max_speed, freq, runing_time, process_time):
    if runing_time < process_time * 2:
        velocity = max_speed * np.sin(2 * np.pi * freq * runing_time)
        d.ctrl[4] = velocity
    runing_time += m.opt.timestep * 5

    return runing_time

def plot(time_log, cmd_pos_log, actual_pos_log, torque_log):
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(time_log, cmd_pos_log, label='Commanded Position', color='green', linestyle='--')
    plt.plot(time_log, actual_pos_log, label='Actual Position', color='blue')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (rad)')
    plt.title('Commanded vs Actual Position')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(time_log, torque_log, label='Torque', color='red')
    plt.xlabel('Time (s)')
    plt.ylabel('Torque (N·m)')
    plt.title('Torque')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_wheel_motor(time_log, cmd_vel_log, actual_vel_log, torque_log):
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(time_log, cmd_vel_log, label='Commanded velocity', color='green', linestyle='--')
    plt.plot(time_log, actual_vel_log, label='Actual velocity', color='blue')
    plt.xlabel('Time (s)')
    plt.ylabel('velocity (rad/s)')
    plt.title('Commanded vs Actual velocity')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(time_log, torque_log, label='Torque', color='green')
    plt.xlabel('Time (s)')
    plt.ylabel('Torque (N·m)')
    plt.title('Torque')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def main(args):
    process_time = 8
    test_radian_max = 1.5
    test_radian_min = 0
    running_time = 0
    log_time = 0

    max_velocity = 50
    frequency = 0.125
    
    # 用于记录数据
    time_log = []
    cmd_pos_log = []
    actual_pos_log = []
    cmd_vel_log = []
    actual_vel_log = []
    torque_log = []

    mujoco.mj_resetDataKeyframe(m, d, 0)  # 0 是關鍵幀的索引
    mujoco.mj_forward(m, d)
    
    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Run simulation for one cycle of process time
        while viewer.is_running():
            if args.part == 'thigh':
                running_time = run_thigh(running_time, process_time, test_radian_max, test_radian_min)
                log_time += m.opt.timestep * 5
                time_log.append(log_time)
                cmd_pos_log.append(d.ctrl[2])
                actual_pos_log.append(d.sensordata[2])
                torque_log.append(d.sensordata[12])
                end_time = process_time*2
            elif args.part == 'calf':
                running_time = run_calf(running_time, process_time, -test_radian_max, test_radian_min)
                log_time += m.opt.timestep * 5
                time_log.append(log_time)
                cmd_pos_log.append(d.ctrl[3])
                actual_pos_log.append(d.sensordata[3])
                torque_log.append(d.sensordata[13])
                end_time = process_time*2
            elif args.part == 'wheel':
                running_time = test_wheel(max_velocity, frequency, running_time, process_time)
                time_log.append(running_time)
                cmd_vel_log.append(d.ctrl[4])
                actual_vel_log.append(d.sensordata[8])
                torque_log.append(d.sensordata[14])
                end_time = process_time * 2 
            if running_time > end_time:
                break
            
            # 执行一步模拟
            step_start = time.time()
            for i in range(5):
                mujoco.mj_step(m, d)
            # 更新渲染
            viewer.sync()
            # 同步时间
            time_until_next_step = m.opt.timestep * 5 - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
        
        # Close the viewer
        viewer.close()

    # Plot results after simulation
    if args.part in ['thigh', 'calf']:
        plot(time_log, cmd_pos_log, actual_pos_log, torque_log)
    else:  # wheel
        plot_wheel_motor(time_log, cmd_vel_log, actual_vel_log, torque_log)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test specific parts of the robot in MuJoCo.")
    parser.add_argument('--part', type=str, choices=['thigh', 'calf', 'wheel'], default='wheel',
                        help="Part of the robot to test: 'thigh', 'calf', or 'wheel'.")
    args = parser.parse_args()
    main(args)