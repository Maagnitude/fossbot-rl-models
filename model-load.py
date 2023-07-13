from stable_baselines3 import PPO
from grid_env import GridEnvironment

# Load the trained agent
model = PPO.load("ppo_agent_")

# Create the environment
env = GridEnvironment()

# Play episodes using the trained agent
obs = env.reset()
done = False

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)
    env.render()