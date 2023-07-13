import gym
from gym import spaces
import numpy as np

class GridEnvironment(gym.Env):
    def __init__(self):
        self.grid_size = 5
        self.obstacle_value = 1
        self.free_block_value = 0

        self.action_space = spaces.Discrete(4)  # Up, Down, Left, Right
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.grid_size, self.grid_size), dtype=np.float32)

        self.reset()

    def reset(self):
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.agent_pos = (0, 0)
        self.goal_pos = (self.grid_size - 1, self.grid_size - 1)

        self.grid[self.goal_pos] = self.free_block_value
        self.grid[self.agent_pos] = self.free_block_value
        
        # Randomly place obstacles
        num_obstacles = int(self.grid_size * self.grid_size * 0.2)  # 20% of grid cells as obstacles
        obstacle_positions = np.random.choice(
            range(self.grid_size * self.grid_size), size=num_obstacles, replace=False
        )
        self.grid = self.grid.flatten()
        self.grid[obstacle_positions] = self.obstacle_value
        self.grid = self.grid.reshape((self.grid_size, self.grid_size))

        return self.grid.copy()

    def step(self, action):
        x, y = self.agent_pos

        if action == 0:  # Up
            x -= 1
        elif action == 1:  # Down
            x += 1
        elif action == 2:  # Left
            y -= 1
        elif action == 3:  # Right
            y += 1

        # Check if the new position is valid
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size and self.grid[x, y] != self.obstacle_value:
            self.agent_pos = (x, y)

        done = self.agent_pos == self.goal_pos
        reward = 1.0 if done else 0.0

        self.grid[self.agent_pos] = self.free_block_value

        return self.grid.copy(), reward, done, {}

    def render(self):
        print(self.grid)

# Example usage
env = GridEnvironment()
observation = env.reset()

done = False
while not done:
    action = env.action_space.sample()
    observation, reward, done, _ = env.step(action)
    env.render()