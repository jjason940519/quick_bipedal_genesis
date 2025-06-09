import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--exp_name", type=str, default="quick_wheel-legged-walking-v13")
args = parser.parse_args()
env_cfg, obs_cfg, reward_cfg, command_cfg, curriculum_cfg, domain_rand_cfg, terrain_cfg, train_cfg = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))
print(obs_cfg)
print(reward_cfg)
print(command_cfg)
print(curriculum_cfg)
print(domain_rand_cfg)
# print(terrain_cfg)
print(train_cfg)
print(env_cfg)