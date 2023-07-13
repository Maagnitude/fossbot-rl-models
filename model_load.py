from stable_baselines3 import PPO
from grid_env import GridEnvironment, REWARD, HIT_OBSTACLE
import os
from datetime import datetime

# Create the environment
env = GridEnvironment()

models_dir = "models/PPO"
model_name = "ppo_agent_80000_new"
model_path = f"{models_dir}/{model_name}"

# Load the trained agent
model = PPO.load(model_path, env=env)

episodes = 10
total_rewards = 0

for ep in range(episodes):
    obs = env.reset()
    done = False
    episode_reward = 0
    REWARD = 0
    
    done = False
    action_list = []
    
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        action_list.append(action)
        env.render()
        
        # print(f"Action list: {action_list}")
        print(f"Episode reward: {episode_reward}")
        print(f"Hit obstacle {HIT_OBSTACLE} times!")
        
    total_rewards += episode_reward

average_reward = total_rewards / episodes

print(f"Average reward over {episodes} episodes: {average_reward}")

if not os.path.exists("records"):
    os.makedirs("records")

current_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
file_path = os.path.join("records", "ppo_records.txt")
with open(file_path, "a") as f:
    f.write(f"{current_time} - {model_name}'s average reward over {episodes} episodes: {average_reward}\n")

# Save the trained agent
model.save(models_dir + f"/{model_name}_new")