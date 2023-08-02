import numpy as np
from stable_baselines3 import PPO
from grid_env import GridEnvironment
import os
import torch as th
from datetime import datetime

models_dir = "models/PPO"
logdir = "logs"
model_name = "ppo_gridworld_9x9"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    
if not os.path.exists(logdir):
    os.makedirs(logdir)
    
# Create the MatrixEnvironment
env = GridEnvironment()

# Set the random seed for reproducibility
np.random.seed(42)

policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=dict(pi=[64, 32, 32, 32, 12], vf=[64, 32, 32, 32, 12]))

model = PPO(policy="MlpPolicy",
            env=env,  
            verbose=1,          
            learning_rate=1e-3,
            policy_kwargs=policy_kwargs,
            tensorboard_log=logdir,
            gamma=0.95)
    
desired_avg_reward = 1000  # Set the desired threshold to 0.0

last_score = -1000
while True:
    model.learn(total_timesteps=10000, reset_num_timesteps=False, tb_log_name="PPO")  # Perform a training iteration

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
            env.moves += 1
            if done == True:
                if reward == 1000.0:
                    print("Goal Reached")
                elif reward == -55.0:
                    print("Crushed in obstacle")
                elif reward == -100.0:
                    print("Hit the boundary")
            #print(obs)
            total_reward += reward
            env.render()            
        print(f"Total moves: {env.moves}")
        print(f"Total Reward: {total_reward}\n")
        episode_rewards.append(total_reward)
    
    # Calculate the average reward over the evaluation episodes
    avg_reward = np.mean(episode_rewards)
    
    if not os.path.exists("records"):
        os.makedirs("records")

    current_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    file_path = os.path.join("records", "ppo_records.txt")
    with open(file_path, "a") as f:
        f.write(f"{current_time} - {model_name}'s average reward over {eval_episodes} episodes: {avg_reward}\n")

    # Print the average reward
    print(f"Average reward: {avg_reward}")
    if avg_reward > last_score:
        last_score = avg_reward
        model.save(f"{models_dir}/{model_name}")

    # Check if the average reward meets the desired threshold
    if avg_reward >= desired_avg_reward:
        print("Training completed. Desired average reward achieved.")
        model.save(f"{models_dir}/{model_name}")
        break