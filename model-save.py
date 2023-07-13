import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from grid_env import GridEnvironment
import os

models_dir = "models/PPO"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    
if not os.path.exists(logdir):
    os.makedirs(logdir)
    
# Create the MatrixEnvironment
env = GridEnvironment()

# Set the random seed for reproducibility
np.random.seed(42)

# Train the PPO agent
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)

TIMESTEPS = 10000
for i in range(1, 10):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{models_dir}/ppo_agent_{TIMESTEPS*i}")