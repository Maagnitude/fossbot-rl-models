from stable_baselines3 import PPO
from grid_env import GridEnvironment
import os
from datetime import datetime
import pygame
import numpy as np

# Create the environment
env = GridEnvironment()

models_dir = "models/PPO"
model_name = "ppo_gridworld"
model_path = f"{models_dir}/{model_name}"

# Load the trained agent
model = PPO.load(model_path, env=env)

desired_avg_reward = 950  # Set the desired threshold to 0.0

last_score = -1600
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

# episodes = 5
# total_rewards = 0

# for ep in range(episodes):
#     obs = env.reset()
#     done = False
#     episode_reward = 0   
    
#     done = False
    
#     while not done:
#         action, _ = model.predict(obs)
#         obs, reward, done, info = env.step(action)
#         episode_reward += reward
#         env.render()
#         pygame.time.delay(30)  # Adjust the speed of visualization (30 milliseconds)
        
#         # INFO
#         print(f"Total moves: {env.moves}")
#         print(f"Action taken: {env.actions[action]}")
#         if env.stand_still == False:
#             print(f"Agent moved " + ("closer to" if env.closer else "away from") + " the goal!")
#         else:
#             print(f"Agent stood still!")
#         print(f"Episode reward: {episode_reward}")
#         print(f"Hit obstacle {env.hit_obstacle} times!\n")
        
#     total_rewards += episode_reward

# average_reward = total_rewards / episodes

# print(f"Average reward over {episodes} episodes: {average_reward}")

# if not os.path.exists("records"):
#     os.makedirs("records")

# current_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
# file_path = os.path.join("records", "ppo_records.txt")
# with open(file_path, "a") as f:
#     f.write(f"{current_time} - {model_name}'s average reward over {episodes} episodes: {average_reward}\n")

# # Save the trained agent
# model.save(models_dir + f"/{model_name}_new")

# # Close the Pygame window when the script ends
# pygame.quit()