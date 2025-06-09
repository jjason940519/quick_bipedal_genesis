import argparse
import os
import pickle
import shutil

from sympy import true

from quick_wheel_legged_env import WheelLeggedEnv
from rsl_rl.runners import OnPolicyRunner

import genesis as gs # type: ignore

def get_train_cfg(exp_name, max_iterations):

    train_cfg_dict = {
        "algorithm": {
            "clip_param": 0.2,
            "desired_kl": 0.006,
            "entropy_coef": 0.01,
            "gamma": 0.995,
            "lam": 0.95,
            "learning_rate": 0.5e-3,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 5,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128, 64],
            "critic_hidden_dims": [256, 128, 64],
            "init_noise_std": 4.0,
        },
        "runner": {
            "algorithm_class_name": "PPO",
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "num_steps_per_env": 25,    #每轮仿真多少step
            "policy_class_name": "ActorCritic",
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
            "runner_class_name": "runner_class_name",
            "save_interval": 100,
        },
        "runner_class_name": "OnPolicyRunner",
        "seed": 1,
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_actions": 6,

        "dof_names": [
            "R_thigh_joint", "R_calf_joint",
            "L_thigh_joint", "L_calf_joint",
            "R_wheel_joint", "L_wheel_joint",
        ],
        "default_joint_angles": {
            # "R_hip_joint": 0.0,       # Fixed, included for completeness
            "R_thigh_joint": 1.22,
            "R_calf_joint": -1.92,
            # "L_hip_joint": 0.0,       # Fixed, included for completeness
            "L_thigh_joint": 1.22,
            "L_calf_joint": -1.92,
            "R_wheel_joint": 0.0,
            "L_wheel_joint": 0.0,
        },
        "dof_limit": {
            # "R_hip_joint": [0.0, 0.0],       # Fixed, no range
            "R_thigh_joint": [0, 2.1],
            "R_calf_joint": [-3, 0],
            # "L_hip_joint": [0.0, 0.0],       # Fixed, no range
            "L_thigh_joint": [0, 2.1],  # Example range, adjust as needed
            "L_calf_joint": [-3, 0],         # Example range
            "R_wheel_joint": [0.0, 0.0],
            "L_wheel_joint": [0.0, 0.0],     # Continuous, no limits enforced
        },
        "safe_force": {
            "R_hip_joint": 0.0,      # Fixed, no force needed
            "R_thigh_joint": 20.0,
            "R_calf_joint": 20.0,
            "L_hip_joint": 0.0,      # Fixed, no force needed
            "L_thigh_joint": 20.0,
            "L_calf_joint": 20.0,
            "R_wheel_joint": 4,
            "L_wheel_joint": 4,
            
        },
        "base_init_pos": {"urdf": [0.0, 0.0, 0.3]},  # Adjusted height (see below)
        "joint_kp": 15,
        "joint_kd": 0.5,
        "wheel_kp": 0.0,
        "wheel_kd": 0.3,
        "thigh_damping": 0.1,
        "thigh_stiffness": 0,
        "thigh_armature": 0.01,
        "calf_damping": 0.1,
        "calf_stiffness": 0,
        "calf_armature": 0.01,
        "wheel_damping": 0.003,
        "wheel_stiffness": 0,
        "wheel_armature": 0.005,
        "termination_if_roll_greater_than": 20,  # Degrees
        "termination_if_pitch_greater_than": 25,
        "termination_if_base_connect_plane_than": True,
        "connect_plane_links": [
            "base_link", "L_thigh_link" ,"L_calf_link", "R_thigh_link", "R_calf_link"
        ],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 15.0,
        "resampling_time_s": 5.0,
        "joint_action_scale": 0.25, #0.5
        "wheel_action_scale": 5,
        "clip_actions": 100.0,
    }
    obs_cfg = {
        # num_obs = num_slice_obs + history_num * num_slice_obs
        "num_obs": 156, #在rsl-rl中使用的变量为num_obs表示state数量
        "num_slice_obs": 26,
        "num_privileged_obs": 165,
        "history_length": 5,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.5,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
            "height_measurements": 5.0,
            "torque": 0.05
        },
        "noise":{
            "use": True,
            "noise_level": 1,
            "lin_vel": 0.1,
            "ang_vel": 0.2,
            "dof_pos": 0.01,
            "dof_vel": 1.5,
            "gravity": 0.05,
            "height_measurements": 0.1,
        }
    }
    # 名字和奖励函数名一一对应
    reward_cfg = {
        "tracking_lin_sigma": 0.10, 
        "tracking_ang_sigma": 0.3,
        "tracking_height_sigma": 0.008, 
        "tracking_similar_legged_sigma": 0.2, #千萬不要隨便亂調
        "tracking_gravity_sigma": 0.018,
        # "tracking_similar_legged_vel_sigma": 0.001,
        "reward_scales": {
            "tracking_lin_vel": 1.85,
            "tracking_ang_vel": 2.5,
            "tracking_base_height": 3,    #和similar_legged对抗，similar_legged先提升会促进此项
            "lin_vel_z": -0.015, #大了影响高度变换速度
            "joint_action_rate": -0.005,
            "wheel_action_rate": -0.002,
            "similar_to_default": 0.0,
            "projected_gravity": 5,
            "similar_legged": 1,  #tracking_base_height和knee_height对抗
            # "similar_legged_vel": 1.5,
            "dof_vel": -0.005,
            "dof_acc": -0.35e-9,
            "dof_force": -0.0002,
            "knee_height": -0.02,    #相当有效，和similar_legged结合可以抑制劈岔和跪地重启，稳定运行
            "ang_vel_xy": -0.015,
            "collision": -0.0008,  #base接触地面碰撞力越大越惩罚，数值太大会摆烂
            # "terrain":0.1,
        },
    }
    command_cfg = {
        "num_commands": 4,
        "base_range": 1.0,  #基础范围
        "lin_vel_x_range": [-1.0, 1.0], #修改范围要调整奖励权重
        "lin_vel_y_range": [-0.0, 0.0],
        "ang_vel_range": [-3.14, 3.14],   #修改范围要调整奖励权重
        "height_target_range": [0.18, 0.32],   #lower会导致跪地
    }
    # 课程学习，奖励循序渐进 待优化
    curriculum_cfg = {
        "curriculum_lin_vel_step":0.001,   #比例
        "curriculum_ang_vel_step":0.0006,   #比例
        "curriculum_height_target_step":0.015,   #高度，先高再低，base_range表示[min+0.7height_range,max]
        "curriculum_lin_vel_min_range":0.25,   #比例
        "curriculum_ang_vel_min_range":0.075,   #比例
        "lin_vel_err_range":[0.1,0.2],  #课程误差阈值
        "ang_vel_err_range":[0.3,0.4],  #课程误差阈值 连续曲线>方波>不波动
    }
    #域随机化 friction_ratio是范围波动 mass和com是偏移波动
    domain_rand_cfg = { 
        "friction_ratio_range":[0.1 , 2.0],
        "random_base_mass_shift_range":[-1 , 1], #质量偏移量
        "random_other_mass_shift_range":[-0.1, 0.1],  #质量偏移量
        "random_base_com_shift":0.05, #位置偏移量 xyz
        "random_other_com_shift":0.01, #位置偏移量 xyz
        "random_KP":[0.9, 1.1], #比例
        "random_KD":[0.9, 1.1], #比例
        "random_default_joint_angles":[-0.05,0.05], #rad
        "randomize_action_delay": True,
        "delay_ms_range": [0, 20],
        "dof_damping_range":[0.8 , 1.2], #比例
        "dof_stiffness_range":[0.8 , 1.2], #比例 
        "dof_armature_range":[0.8 , 1.2], #比例 额外惯性 类似电机减速器惯性
    }
    #地形配置
    terrain_cfg = {
        "terrain":True, #是否开启地形
        "train":"agent_train_gym",
        "eval":"agent_eval_gym",    # agent_eval_gym/circular
        "num_respawn_points":3,
        "respawn_points":[
            [-5.0, -5.0, 0.0],    #plane地形坐标，一定要有，为了远离其他地形
            [5.0, 5.0, 0.0],
            [15.0, 5.0, 0.08],
        ],
        "horizontal_scale":0.1,
        "vertical_scale":0.001,
    }
    return env_cfg, obs_cfg, reward_cfg, command_cfg, curriculum_cfg, domain_rand_cfg, terrain_cfg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="quick_wheel-legged-walking-v14")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=10000)
    args = parser.parse_args()

    gs.init(logging_level="warning",backend=gs.gpu)
    gs.device="cuda:0"
    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, curriculum_cfg, domain_rand_cfg, terrain_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    env = WheelLeggedEnv(
        num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, 
        command_cfg=command_cfg, curriculum_cfg=curriculum_cfg, 
        domain_rand_cfg=domain_rand_cfg, terrain_cfg=terrain_cfg
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, curriculum_cfg, domain_rand_cfg, terrain_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()

"""
# training
python examples/locomotion/go2_train.py
"""
