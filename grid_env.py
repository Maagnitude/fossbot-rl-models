import gym
from gym import spaces
import numpy as np
from visualization import visualize_grid

def manhattan(a, b):
    return sum(abs(val1-val2) for val1, val2 in zip(a,b))

class GridEnvironment(gym.Env):
    def __init__(self):
        self.grid_size = 9
        self.agent_value = 2
        self.obstacle_value = 1
        self.free_block_value = 0
        self.goal_value = 3
        self.reward = 0
        self.hit_obstacle = 0
        self.closer = False
        self.stand_still = False
        self.moves = 0
        
        self.actions = ["turn_left", "turn_right", "move_forward"]
        self.num_actions = len(self.actions)

        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.grid_size, self.grid_size), dtype=np.float32)

        self.reset()

    def reset(self):
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.agent_pos = (0, 0)
        self.agent_orientation = 1  # 0 for NORTH, 1 for EAST, 2 for SOUTH, 3 for WEST
        self.goal_pos = (self.grid_size - 1, self.grid_size - 1)
        self.reward = 0
        self.stand_still = False
        self.closer = False
        self.moves = 0
        self.goal_from_agent = (2, 1)

        self.grid[self.goal_pos] = self.goal_value
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
        yaxis_orientation, xaxis_orientation = self.goal_from_agent
        x, y = self.agent_pos
        old_position = self.agent_pos

        if action == 0:  # turn_left
            self.agent_orientation = (self.agent_orientation - 1) % 4
            self.stand_still = True
        elif action == 1:  # turn_right
            self.agent_orientation = (self.agent_orientation + 1) % 4
            self.stand_still = True
        elif action == 2:  # move_forward
            if self.agent_orientation == 0:  # NORTH
                x -= 1
            elif self.agent_orientation == 1:  # EAST
                y += 1
            elif self.agent_orientation == 2:  # SOUTH
                x += 1
            elif self.agent_orientation == 3:  # WEST
                y -= 1   
            self.stand_still = False  
          
        # OLD ACTIONS        
        # if action == 0:  # Up
        #     x -= 1
        # elif action == 1:  # Down
        #     x += 1
        # elif action == 2:  # Left
        #     y -= 1
        # elif action == 3:  # Right
        #     y += 1

        # Check if the new position is valid
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size and self.grid[x, y] != self.obstacle_value and self.stand_still == False:
            self.grid[self.agent_pos] = self.free_block_value
            self.agent_pos = (x, y)
            self.grid[self.agent_pos] = self.agent_value
            if self.agent_orientation in self.goal_from_agent:
                self.reward = 60.0
            else:
                self.reward = 30.0
        elif self.stand_still == True and (self.agent_orientation in self.goal_from_agent):
            # Agent is facing the goal
            self.grid[self.agent_pos] = self.agent_value
            self.reward = -0.4
        elif self.stand_still == True and (self.agent_orientation not in self.goal_from_agent):  
            # Agent is not facing the goal   
            self.grid[self.agent_pos] = self.agent_value
            self.reward = -1.0
        else:
            # Agent hit an obstacle
            self.grid[self.agent_pos] = self.agent_value
            self.reward = -2.0
            self.hit_obstacle += 1
        
        self.moves += 1
        done = self.agent_pos == self.goal_pos
        if done:
            self.reward = 1000.0
        elif manhattan(self.goal_pos, self.agent_pos) < manhattan(self.goal_pos, old_position):
            # Reward agent for moving closer to goal
            self.closer = True
            self.reward = (self.grid_size*2 - manhattan(self.goal_pos, self.agent_pos))
            if self.agent_orientation == 0 or self.agent_orientation == 2:
                yaxis_orientation = self.agent_orientation
            if self.agent_orientation == 1 or self.agent_orientation == 3:
                xaxis_orientation = self.agent_orientation
            self.goal_from_agent = (yaxis_orientation, xaxis_orientation)
        elif manhattan(self.goal_pos, self.agent_pos) > manhattan(self.goal_pos, old_position):
            # Penalize agent for moving away from goal
            self.reward = -(self.grid_size*2 - manhattan(self.goal_pos, self.agent_pos))
            self.closer = False
            if self.agent_orientation == 0 or self.agent_orientation == 2:
                yaxis_orientation = self.agent_orientation
            if self.agent_orientation == 1 or self.agent_orientation == 3:
                xaxis_orientation = self.agent_orientation
            self.goal_from_agent = (yaxis_orientation, xaxis_orientation)
        else:
            self.closer = False

        return self.grid.copy(), self.reward, done, {}

    def render(self):
        # print(self.grid)
        visualize_grid(self.grid, self.agent_pos, self.agent_orientation)
        
if __name__ == "__main__":
    env = GridEnvironment()
    env.reset()
    env.render()