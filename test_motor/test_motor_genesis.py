import numpy as np
from sympy import true
import genesis as gs
import matplotlib.pyplot as plt
import argparse

gs.init(backend=gs.gpu)

scene = gs.Scene(
    viewer_options = gs.options.ViewerOptions(
        camera_pos    = (0, -3.5, 2.5),
        camera_lookat = (0.0, 0.0, 0.5),
        camera_fov    = 30,
        max_FPS       = 60,
    ),
    sim_options = gs.options.SimOptions(
        dt = 0.01,
    ),
    show_viewer = True,
)
plane = scene.add_entity(
    gs.morphs.Plane(),
)
robot = scene.add_entity(
    gs.morphs.URDF(
        file = "../assets/quick_bipedal_urdf/urdf/quick_bipedal.urdf",
        # file = "../assets/wheel_legged_urdf/urdf/bi_urdf.urdf",
        pos = [0.0, 0.0, 0.9],
        fixed = True,
        convexify = True
    )
)
scene.build()

jnt_names = [
    "L_thigh_joint", "L_calf_joint",
    "R_thigh_joint", "R_calf_joint",
    "L_wheel_joint", "R_wheel_joint"
]
dofs_idx = [robot.get_joint(name).dof_idx_local for name in jnt_names]

print(dofs_idx)

robot.set_dofs_kp(
    kp             = np.array([15, 15, 15, 15, 0, 0]),
    dofs_idx_local = dofs_idx,
)
robot.set_dofs_kv(
    kv             = np.array([0.5, 0.5, 0.5, 0.5, 0.3, 0.3]),
    dofs_idx_local = dofs_idx,
)
robot.set_dofs_force_range(
    lower          = np.array([-13, -13, -13, -13, -4, -4]),
    upper          = np.array([13, 13, 13, 13, 4, 4]),
    dofs_idx_local = dofs_idx,
)

robot.set_dofs_damping(
    damping        = np.array([0.1, 0.1, 0.1, 0.1, 0.003, 0.003]),
    dofs_idx_local = dofs_idx,
)
robot.set_dofs_armature(
    armature       = np.array([0.01 ,0.01, 0.01, 0.01, 0.005, 0.005]),
    dofs_idx_local = dofs_idx,
)

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

def run_thigh(runing_time, process_time, test_radian_max, test_radian_min):
    if runing_time < process_time:
        runing_time += 0.01
        phase = np.tanh(runing_time / 3)
        pos = phase * test_radian_max + (1 - phase) * test_radian_min
        robot.control_dofs_position(position = np.array([pos,0]), dofs_idx_local = dofs_idx[0:2])
    elif runing_time < process_time * 2:
        runing_time += 0.01
        phase = np.tanh((runing_time - process_time) / 3)
        pos = phase * test_radian_min + (1 - phase) * test_radian_max
        robot.control_dofs_position(position = np.array([pos,0]), dofs_idx_local = dofs_idx[0:2])

    return runing_time, pos

def run_calf(runing_time, process_time, test_radian_max, test_radian_min):
    if runing_time < process_time:
        runing_time += 0.01
        phase = np.tanh(runing_time / 3)
        pos = phase * test_radian_max + (1 - phase) * test_radian_min
        robot.control_dofs_position(position = np.array([0,pos]), dofs_idx_local = dofs_idx[0:2])
    else:
        runing_time += 0.01
        phase = np.tanh((runing_time - process_time) / 3)
        pos = phase * test_radian_min + (1 - phase) * test_radian_max
        robot.control_dofs_position(position = np.array([0,pos]), dofs_idx_local = dofs_idx[0:2])

    return runing_time, pos

def test_wheel(max_speed, freq, runing_time, process_time):
    if runing_time < process_time * 2:
        vel = max_speed * np.sin(2 * np.pi * freq * runing_time)
        robot.control_dofs_velocity(velocity = np.array([vel]) ,dofs_idx_local = dofs_idx[4])
    runing_time += 0.01   
    return runing_time, vel

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

    end_time = process_time*2

    while true:
        if args.part == 'wheel':
            running_time, comm_v = test_wheel(max_velocity, frequency, running_time, process_time)
            time_log.append(running_time)
            cmd_vel_log.append(comm_v)
            actu_v = robot.get_dofs_velocity(dofs_idx_local=dofs_idx[4]).cpu().numpy()
            actual_vel_log.append(actu_v)
            actu_t = robot.get_dofs_control_force(dofs_idx_local=dofs_idx[4]).cpu().numpy()
            torque_log.append(actu_t)    
        elif args.part == 'calf':
            running_time, p = run_calf(running_time, process_time, -test_radian_max, test_radian_min)
            log_time += 0.01
            time_log.append(log_time)
            cmd_pos_log.append(p)
            actu_p = robot.get_dofs_position(dofs_idx_local=dofs_idx[1]).cpu().numpy()
            actual_pos_log.append(actu_p)
            actu_t = robot.get_dofs_control_force(dofs_idx_local=dofs_idx[1]).cpu().numpy()
            torque_log.append(actu_t)
        elif args.part == 'thigh':
            running_time, p = run_thigh(running_time, process_time, test_radian_max, test_radian_min)
            log_time += 0.01
            time_log.append(log_time)
            cmd_pos_log.append(p)
            actu_p = robot.get_dofs_position(dofs_idx_local=dofs_idx[0]).cpu().numpy()
            actual_pos_log.append(actu_p)
            actu_t = robot.get_dofs_control_force(dofs_idx_local=dofs_idx[0]).cpu().numpy()
            torque_log.append(actu_t)
        scene.step()
        if running_time > end_time:
            break

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