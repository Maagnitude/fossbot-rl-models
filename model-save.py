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
for i in range(1, 30):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{models_dir}/ppo_agent_{TIMESTEPS*i}")

# # Evaluate the trained agent
# n_eval_episodes = 10
# total_rewards = 0

# for _ in range(n_eval_episodes):
#     obs = env.reset()
#     done = False
#     episode_reward = 0

#     while not done:
#         action, _ = model.predict(obs, deterministic=True)
#         obs, reward, done, _ = env.step(action)
#         episode_reward += reward

#     total_rewards += episode_reward

# average_reward = total_rewards / n_eval_episodes

# print(f"Average reward over {n_eval_episodes} episodes: {average_reward}")

# os.makedirs("models", exist_ok=True)

# # Save the trained agent
# model.save("models/ppo_agent")