import gym
from gym import spaces
import numpy as np

def manhattan(a, b):
    return sum(abs(val1-val2) for val1, val2 in zip(a,b))

class GridEnvironment(gym.Env):
    def __init__(self):
        self.grid_size = 9
        self.agent_value = 2
        self.obstacle_value = 1
        self.free_block_value = 0
        self.reward = 0
        self.hit_obstacle = 0
        self.closer = False

        self.action_space = spaces.Discrete(4)  # Up, Down, Left, Right
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.grid_size, self.grid_size), dtype=np.float32)

        self.reset()

    def reset(self):
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.agent_pos = (0, 0)
        self.goal_pos = (self.grid_size - 1, self.grid_size - 1)
        self.reward = 0

        self.grid[self.goal_pos] = self.free_block_value
        self.grid[self.agent_pos] = self.free_block_value
        
        # Randomly place obstacles
        num_obstacles = int(self.grid_size * self.grid_size * 0.15)  # 30% of grid cells as obstacles
        obstacle_positions = np.random.choice(
            range(self.grid_size * self.grid_size), size=num_obstacles, replace=False
        )
        # Make sure the goal is not an obstacle
        obstacle_positions = obstacle_positions[obstacle_positions != self.grid_size * self.grid_size - 1]
        
        # Make sure that the position before the goal is not an obstacle
        obstacle_positions = obstacle_positions[obstacle_positions != self.grid_size * self.grid_size - 2]
        
        # Make sure that the position after the first position is not an obstacle
        obstacle_positions = obstacle_positions[obstacle_positions != 1]
        
        self.grid = self.grid.flatten()
        self.grid[obstacle_positions] = self.obstacle_value
        
        self.grid = self.grid.reshape((self.grid_size, self.grid_size))

        return self.grid.copy()

    def step(self, action):
        x, y = self.agent_pos
        old_position = self.agent_pos

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
            self.grid[self.agent_pos] = self.free_block_value
            self.agent_pos = (x, y)
            self.grid[self.agent_pos] = self.agent_value
        else:
            self.grid[self.agent_pos] = self.free_block_value
            self.grid[self.agent_pos] = self.agent_value
            self.reward -= manhattan(self.goal_pos, self.agent_pos)
            self.hit_obstacle += 1

        done = self.agent_pos == self.goal_pos
        if done:
            self.reward += 10000000.0
        elif manhattan(self.goal_pos, self.agent_pos) < manhattan(self.goal_pos, old_position):
            # Reward agent for moving closer to goal
            self.closer = True
            self.reward += (30 - manhattan(self.goal_pos, self.agent_pos))
        elif manhattan(self.goal_pos, self.agent_pos) > manhattan(self.goal_pos, old_position):
            # Penalize agent for moving away from goal
            self.reward -= manhattan(self.goal_pos, self.agent_pos)
            self.closer = False

        return self.grid.copy(), self.reward, done, {}

    def render(self):
        print(self.grid)

# Example usage
# env = GridEnvironment()
# observation = env.reset()

# done = False
# action_list = []
# while not done:
#     action = env.action_space.sample()
#     observation, reward, done, _ = env.step(action)
#     action_list.append(action)
#     env.render()

# print(f"Action list: {action_list}")
# print(f"Final reward: {reward}")
# print(f"Hit obstacle {HIT_OBSTACLE} times!")