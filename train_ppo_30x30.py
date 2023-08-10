import numpy as np
from stable_baselines3 import PPO
from grid_env_30x30_sim import GridEnvironment
import os
import torch as th
from datetime import datetime
import wandb

wandb.init(
    project="grid_30x30_models",
    resume=True
)

models_dir = "models/PPO"
model_name = "ppo_gridworld_30x30_sim"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    
# Create the MatrixEnvironment
env = GridEnvironment()

policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=dict(pi=[64, 32, 32, 32, 12], vf=[64, 32, 32, 32, 12]))

model = PPO(policy="MlpPolicy",
            env=env,           
            learning_rate=1e-3,
            policy_kwargs=policy_kwargs,
            batch_size=32,
            gamma=0.95)
    
desired_avg_reward = 1000  # Set the desired threshold to 0.0

last_score = -1000
total_scores = 0
updates = 1
saves = 0
avg_score = 0.0
total_last_action_distance = 0.0

try:
    while True:
        model.learn(total_timesteps=10000, reset_num_timesteps=False, progress_bar=True)  # Perform a training iteration

        # Evaluate the model's performance over multiple episodes
        eval_episodes = 10  # Set the number of evaluation episodes
        episode_rewards = []

        for i in range(eval_episodes):
            obs = env.reset()
            done = False
            total_reward = 0
            total_last_action_distance = 0
            
            print(f"Eval Episode: {i}")
            while not done:
                action, _ = model.predict(obs)
                obs, reward, done, _ = env.step(action)
                if reward == 1000.0:
                    print("Goal Reached")
                elif reward == -15.0 and env.steps == env.max_steps:
                    print("Max steps reached")
                elif reward == -15.0 and (env.spinning_left > 15 or env.spinning_right > 15):
                    print("Spinning too much")
                elif reward == -15.0:
                    print("Hit an obstacle")
                # print(f"Step: {env.steps},   Euclidean Distance: {obs[0]},   Degrees from goal: {env.agent_goal_diff}")
                total_reward += reward
                # env.render()
            total_last_action_distance += obs[0]            
            print(f"Total moves: {env.steps}")
            print(f"Total Reward: {total_reward}\n")
            episode_rewards.append(total_reward)
        
        # Calculate the average reward over the evaluation episodes
        avg_reward = np.mean(episode_rewards)
        avg_distance = total_last_action_distance / eval_episodes
        
        wandb.log({"reward": avg_reward, "avg_distance": avg_distance})
        
        updates += 1

        # Print the average reward
        print(f"Average reward: {avg_reward}")
        if avg_reward > last_score:
            saves += 1
            last_score = avg_reward
            total_scores += last_score
            model.save(f"{models_dir}/{model_name}")
            
        avg_score = total_scores / saves

        # Check if the average reward meets the desired threshold
        if avg_reward >= desired_avg_reward:
            print("Training completed. Desired average reward achieved.")
            model.save(f"{models_dir}/{model_name}")
            break
except KeyboardInterrupt:
    print("Training interrupted.")
finally:
    try:
        ans = input(f"Average score of previous updates: {avg_score}\nLast average reward: {avg_reward}\nDo you want to save it?(Y/n)")
        if ans == "Y" or ans == "y":
            model.save(f"{models_dir}/{model_name}")
            print("Model saved!")
        else:
            print("Model not saved!")
    except NameError:
        model.save(f"{models_dir}/{model_name}")
        print("Model saved!")
    print("Training ended!")
    wandb.finish()