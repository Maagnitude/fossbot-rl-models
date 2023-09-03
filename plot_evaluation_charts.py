import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV files into DataFrames
ppo_csv_file_name = "results/ppo_evaluation_data.csv"
dqn_csv_file_name = "results/dqn_evaluation_data.csv"
df_ppo = pd.read_csv(ppo_csv_file_name)
df_dqn = pd.read_csv(dqn_csv_file_name)

# Extract the last row of data for both PPO and DQN
ppo_last_row = df_ppo.iloc[-1]
dqn_last_row = df_dqn.iloc[-1]

# Define the labels and data for the bar plot
labels = ["Finished Percentage (%)", "Average Reward (x10)", "Average Steps", "Average Total Distance"]
ppo_data = ppo_last_row[1:]  # Exclude the first column (Episode Number) for PPO
dqn_data = dqn_last_row[1:]  # Exclude the first column (Episode Number) for DQN

# Convert percentage values to numeric
ppo_data[0] = float(ppo_data[0].rstrip('%'))
dqn_data[0] = float(dqn_data[0].rstrip('%'))

# Divide the average reward values by 10 to have a maximum value of 100
ppo_data[1] /= 10
dqn_data[1] /= 10

# Set different colors for each feature and each model
colors = ['b', 'orange']
models = ['PPO', 'DQN']
n_features = len(labels)
bar_width = 0.20
index = np.arange(n_features)

# Create a figure and set the bar width
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the bars for PPO
ax.bar(index - bar_width / 2, ppo_data, bar_width, label='PPO', color=colors[0])

# Plot the bars for DQN
ax.bar(index + bar_width / 2, dqn_data, bar_width, label='DQN', color=colors[1])

# Set labels and title
ax.set_xlabel('Metrics')
ax.set_ylabel('Values')
ax.set_title('Comparison of PPO and DQN Metrics (1000 Episodes)')
ax.set_xticks(index)
ax.set_xticklabels(labels)
ax.legend()

# Add labels for the values above the bars
for i, v in enumerate(ppo_data):
    ax.text(i - bar_width / 2, v + 1, str(round(v, 2)), ha='center', va='bottom', color='black')
for i, v in enumerate(dqn_data):
    ax.text(i + bar_width / 2, v + 1, str(round(v, 2)), ha='center', va='bottom', color='black')

# Display the plot
plt.tight_layout()
plt.show()