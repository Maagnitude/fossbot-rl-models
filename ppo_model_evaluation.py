import numpy as np
import csv
from stable_baselines3 import PPO
from final_grid_env import GridWorldEnv
# from generator import image_to_map
# import torch as th

grid_map = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
            [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
            [1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

# grid_map = image_to_map('map.jpg', 10)


env = GridWorldEnv(grid_map, range_gs = True)
#env = DummyVecEnv([lambda: env])


# policy_kwargs = dict(activation_fn=th.nn.ReLU,
#                      net_arch=dict(pi=[64, 32,32,12], vf=[64, 32,32,12]))

# model = PPO(policy="MlpPolicy",
#             env=env,            
#             learning_rate=1e-3,
#             policy_kwargs=policy_kwargs,
#             gamma=0.95)

model = PPO.load("models/PPO/ppo_gridworld_new", env=env)

# Evaluate the model's performance over multiple episodes
num_evaluation_episodes = 1000
total_episodes = []             # 1st row: Episode number
finished = []                   # 2nd row: Finished or not
episode_rewards = []            # 3rd row: Episode reward
episode_total_steps = []        # 4th row: Episode total steps
episode_total_distances = []    # 5th row: Episode total distances

fossbot_trajectories = []
fossbot_positions = []
goal_positions = []


for i in range(1, num_evaluation_episodes + 1):
    obs = env.reset()
    done = False
    episode_total_reward = 0.0
    episode_steps = 0
    episode_total_distance = 0.0
    reached_goal = False
    fossbot_episode_positions = []
    first_iteration = True
    
    print(f"Eval Episode: {i}")
    total_episodes.append(i)
    while not done:
        if first_iteration:
            fossbot_initial_position = env.agent_position
            goal_position = env.goal_position
            first_iteration = False
        fossbot_episode_positions.append(env.agent_position)
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        episode_steps += 1
        if reward == 1000:
            print("Goal Reached")
            finished.append(1)
            reached_goal = True
        
        episode_total_reward += reward
        # env.render("human")            
    # print(f"Total Reward: {total_reward}")
    fossbot_episode_positions.append(env.agent_position) # To append its last position
    fossbot_trajectories.append(fossbot_episode_positions)
    
    if not reached_goal:
        finished.append(0)
        
    for j in range(len(fossbot_episode_positions)-1):
        episode_total_distance += np.linalg.norm(np.array(fossbot_episode_positions[j])-np.array(fossbot_episode_positions[j+1]))
        
    fossbot_positions.append(fossbot_initial_position)
    goal_positions.append(goal_position)    
    episode_total_distances.append(episode_total_distance)
    episode_rewards.append(episode_total_reward)
    episode_total_steps.append(episode_steps)
    
percentage_finished = np.mean(finished) * 100
average_model_reward = np.mean(episode_rewards)
average_model_total_steps = np.mean(episode_total_steps)
average_model_total_distance = np.mean(episode_total_distances)
    
data = [["Episode Number", "Finished", "Reward", "Total Steps", "Total Distance"]]
data.extend(list(zip(total_episodes, finished, episode_rewards, episode_total_steps, episode_total_distances)))
data.append(["", f"{percentage_finished:.2f}%", average_model_reward, average_model_total_steps, average_model_total_distance])


csv_file_name = "ppo_evaluation_data.csv"

with open(csv_file_name, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerows(data)
    
data2 = [["Episode Number", "Fossbot Init Positions", "Goal Positions", "Fossbot Trajectory"]]
data2.extend(list(zip(total_episodes, fossbot_positions, goal_positions, fossbot_trajectories)))
    
csv_file_name2 = "ppo_evaluation_trajectories.csv"
    
with open(csv_file_name2, 'w', newline='') as csvfile2:
    csvwriter = csv.writer(csvfile2)
    csvwriter.writerows(data2)