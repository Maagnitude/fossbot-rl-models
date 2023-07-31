import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from grid_env import GridEnvironment
import os
import torch as th

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

policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=dict(pi=[64, 32, 32, 12], vf=[64, 32, 32, 12]))

model = PPO(policy="MlpPolicy",
            env=env,  
            verbose=1,          
            learning_rate=1e-3,
            policy_kwargs=policy_kwargs,
            tensorboard_log=logdir,
            gamma=0.95)

# Train the PPO agent
# model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)

# TIMESTEPS = 5000
# for i in range(1, 10):
#     model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
#     model.save(f"{models_dir}/ppo_agent_{TIMESTEPS*i}")
    
desired_avg_reward = 1000  # Set the desired threshold to 0.0

last_score = -10000
while True:
    model.learn(total_timesteps=10000, reset_num_timesteps=False)  # Perform a training iteration

    # Evaluate the model's performance over multiple episodes
    eval_episodes = 10  # Set the number of evaluation episodes
    episode_rewards = []

    for i in range(eval_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        
        print(f"Eval Episode: {i}")
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            if reward == 1000:
                print("Goal Reached")
            #print(obs)
            total_reward += reward
            env.render()            
        print(f"Total moves: {env.moves}")
        print(f"Total Reward: {total_reward}")
        print(f"Hit obstacle {env.hit_obstacle} times!\n")
        episode_rewards.append(total_reward)
        
    
    # Calculate the average reward over the evaluation episodes
    avg_reward = np.mean(episode_rewards)

    # Print the average reward
    print(f"Average reward: {avg_reward}")
    if avg_reward > last_score:
        last_score = avg_reward
        model.save("models/PPO/ppo_gridworld")

    # Check if the average reward meets the desired threshold
    if avg_reward >= desired_avg_reward:
        print("Training completed. Desired average reward achieved.")
        model.save("ppo_gridworld")
        break