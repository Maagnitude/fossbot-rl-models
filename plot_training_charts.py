import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a DataFrame
csv_file_name = "results/ppo_dqn_training_results.csv"
df = pd.read_csv(csv_file_name)

# Extract data from the DataFrame
steps = df['Step']
dqn_rewards = df['DQN-30x30-new - reward']
ppo_rewards = df['PPO-30x30-new - reward']

# Create a figure with subplots
fig, ax = plt.subplots(figsize=(12, 6))

# Plot DQN rewards up to step 115
ax.plot(steps[steps <= 115], dqn_rewards[steps <= 115], label='DQN', linestyle='--', linewidth=2)

# Plot PPO rewards up to step 65
ax.plot(steps[steps <= 115], ppo_rewards[steps <= 115], label='PPO', linestyle='-', linewidth=2)

# Set labels and title
plt.xlabel('Evaluation Cycles (x10 Episodes)', fontsize=15)
plt.ylabel('Average Reward', fontsize=15)
plt.title('DQN vs PPO Training Results', fontsize=15)
plt.legend(fontsize=14)
plt.grid(True)

# Show the plot
plt.show()

# # MOVING AVERAGE

# # Define the window size for the moving average
# window_size = 10

# # Calculate the moving average for DQN and PPO rewards
# dqn_ma = dqn_rewards.rolling(window=window_size).mean()
# ppo_ma = ppo_rewards.rolling(window=window_size).mean()

# # Create a figure with subplots
# fig, ax = plt.subplots(figsize=(12, 6))

# # Plot smoothed DQN rewards up to step 115
# ax.plot(steps[steps <= 115], dqn_ma[steps <= 115], label='DQN Reward (Moving Average)', linestyle='-', linewidth=2)

# # Plot smoothed PPO rewards up to step 65
# ax.plot(steps[steps <= 115], ppo_ma[steps <= 115], label='PPO Reward (Moving Average)', linestyle='-', linewidth=2)

# # Set labels and title
# plt.xlabel('Evaluation Cycles (x10 Episodes)')
# plt.ylabel('Average Reward (Moving Average)')
# plt.title('DQN vs PPO Training Results with Moving Average')
# plt.legend()
# plt.grid(True)

# # Show the plot
# plt.show()