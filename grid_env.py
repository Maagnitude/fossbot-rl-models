import gym
from gym import spaces
import numpy as np
from visualization import visualize_grid
import math

def manhattan(a, b):
    return sum(abs(val1-val2) for val1, val2 in zip(a,b))

def goal_direction(agent_pos, goal_pos, agent_orientation):
            # Calculate the updated goal_from_agent in azimuth
            goal_from_agent = math.degrees(math.atan2(goal_pos[1] - agent_pos[1], goal_pos[0] - agent_pos[0]))
            diff = abs(agent_orientation - goal_from_agent - 90)
            if diff > 45:
                reward = -10.0
            else:
                reward = -1.0
            return reward, goal_from_agent

class GridEnvironment(gym.Env):
    def __init__(self):
        self.grid_size = 15
        self.free_block_value = 0
        self.obstacle_value = 1
        self.agent_value = 2
        self.goal_value = 3
        self.agent_on_obstacle_value = 4
        self.agent_on_goal_value = 5
        self.reward = 0
        self.closer = False
        self.stand_still = True
        self.moves = 0
        self.agent_orientation = 90
        self.goal_from_agent = 45
        
        self.actions = ["TurnLeft", "TurnRight", "MoveForward"]
        self.num_actions = len(self.actions)

        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.grid_size, self.grid_size), dtype=np.float32)

        self.reset()

    def reset(self):
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.agent_pos = (1, 1)
        self.agent_orientation = 90
        self.goal_pos = (self.grid_size - 2, self.grid_size - 2)
        self.reward = 0
        self.stand_still = True
        self.closer = False
        self.moves = 0
        self.goal_from_agent = math.degrees(math.atan2(self.goal_pos[1] - self.agent_pos[1], self.goal_pos[0] - self.agent_pos[0]))
        
        # 20% obstacles
        # self.grid = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        #                       [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        #                       [1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
        #                       [1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
        #                       [1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
        #                       [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1],
        #                       [1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
        #                       [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        #                       [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        #                       [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1],
        #                       [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1],
        #                       [1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1],
        #                       [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1],
        #                       [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 1],
        #                       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        
        self.grid = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                              [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                              [1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
                              [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                              [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
                              [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
                              [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                              [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                              [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                              [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                              [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                              [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
                              [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
                              [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 1],
                              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        
        
        # RANDOM OBSTACLES
        # for i in range(self.grid_size):
        #     if i == 0 or i == self.grid_size - 1:
        #         obstacle_positions = [j for j in range(i * self.grid_size, i * self.grid_size + self.grid_size)]
        #     else:
        #         obstacle_positions = [i * self.grid_size, i * self.grid_size + self.grid_size - 1]

        #     for pos in obstacle_positions:
        #         self.grid[pos // self.grid_size, pos % self.grid_size] = self.obstacle_value

        # for j in range(1, self.grid_size - 1):
        #     obstacle_positions = [j * self.grid_size, j * self.grid_size + self.grid_size - 1]
        #     for pos in obstacle_positions:
        #         self.grid[pos // self.grid_size, pos % self.grid_size] = self.obstacle_value
        
        # # Randomly place obstacles
        # num_obstacles = int(self.grid_size * self.grid_size * 0.06)  # 6% of grid cells as obstacles
        # obstacle_positions = np.random.choice(
        #     range(self.grid_size * self.grid_size), size=num_obstacles, replace=False
        # )
        # # Make sure the goal is not an obstacle
        # obstacle_positions = obstacle_positions[obstacle_positions != self.grid_size * self.grid_size - 2]
        
        # # Make sure that the position before the goal is not an obstacle
        # obstacle_positions = obstacle_positions[obstacle_positions != self.grid_size * self.grid_size - 3]
        
        # # Make sure that the position after the first position is not an obstacle
        # obstacle_positions = obstacle_positions[obstacle_positions != 1]
        
        # self.grid = self.grid.flatten()
        # self.grid[obstacle_positions] = self.obstacle_value
        
        # self.grid = self.grid.reshape((self.grid_size, self.grid_size))
        self.grid[self.goal_pos] = self.goal_value
        # self.grid[self.agent_pos] = self.agent_value
        self.grid[self.agent_pos] = self.agent_value

        return self.grid.copy()/5

    def step(self, action):
        x, y = self.agent_pos
        done = False

        if action == 0:  # TurnLeft
            self.agent_orientation = (self.agent_orientation - 90) % 360
            self.stand_still = True
        elif action == 1:  # TurnRight
            self.agent_orientation = (self.agent_orientation + 90) % 360
            self.stand_still = True
        elif action == 2:  # MoveForward
            if self.agent_orientation == 0:  # NORTH
                x -= 1
            elif self.agent_orientation == 90:  # EAST
                y += 1
            elif self.agent_orientation == 180:  # SOUTH
                x += 1
            elif self.agent_orientation == 270:  # WEST
                y -= 1   
            self.stand_still = False  

        # Check if the new position is valid
        if 1 <= x < self.grid_size-1 and 1 <= y < self.grid_size-1:
            if self.grid[x, y] != self.obstacle_value and self.stand_still == False:
                self.grid[self.agent_pos] = self.free_block_value
                self.reward = 0.0
                self.agent_pos = (x, y)
                if self.grid[self.agent_pos] == self.goal_value:
                    self.reward = 1000.0
                    self.grid[self.agent_pos] = self.agent_on_goal_value
                    done = True
                else:
                    self.grid[self.agent_pos] = self.agent_value
                    self.reward, self.goal_from_agent = goal_direction(self.agent_pos, self.goal_pos, self.agent_orientation)       
                    if not (30 <= self.goal_from_agent <= 60):
                        self.reward = -1.0        
            elif self.stand_still == True:
                self.reward, self.goal_from_agent = goal_direction(self.agent_pos, self.goal_pos, self.agent_orientation)
            elif self.grid[x, y] == self.obstacle_value:
                self.grid[self.agent_pos] = self.free_block_value
                self.agent_pos = (x, y)
                self.reward = -65.0
                self.grid[self.agent_pos] = self.agent_on_obstacle_value
                done = True
        else:
            # Agent hit the boundary
            self.grid[self.agent_pos] = self.free_block_value
            if x == self.grid_size-1:
                self.grid[x, y] = self.agent_on_obstacle_value
            elif x == 0:
                self.grid[x, y] = self.agent_on_obstacle_value
            elif y == self.grid_size-1:
                self.grid[x, y] = self.agent_on_obstacle_value
            elif y == 0:
                self.grid[x, y] = self.agent_on_obstacle_value
            self.reward = -100.0
            done = True

        # elif manhattan(self.goal_pos, self.agent_pos) < manhattan(self.goal_pos, old_position):
        #     # Reward agent for moving closer to goal
        #     self.closer = True
        #     # self.reward = (self.grid_size*2 - manhattan(self.goal_pos, self.agent_pos))
        #     if self.agent_orientation == 0 or self.agent_orientation == 2:
        #         yaxis_orientation = self.agent_orientation
        #     if self.agent_orientation == 1 or self.agent_orientation == 3:
        #         xaxis_orientation = self.agent_orientation
        #     self.goal_from_agent = (yaxis_orientation, xaxis_orientation)
        # elif manhattan(self.goal_pos, self.agent_pos) > manhattan(self.goal_pos, old_position):
        #     # Penalize agent for moving away from goal
        #     # self.reward = -(self.grid_size*2 - manhattan(self.goal_pos, self.agent_pos))
        #     self.closer = False
        #     if self.agent_orientation == 0 or self.agent_orientation == 2:
        #         yaxis_orientation = self.agent_orientation
        #     if self.agent_orientation == 1 or self.agent_orientation == 3:
        #         xaxis_orientation = self.agent_orientation
        #     self.goal_from_agent = (yaxis_orientation, xaxis_orientation)
        # else:
        #     self.closer = False
        self.goal_from_agent = math.degrees(math.atan2(self.goal_pos[1] - self.agent_pos[1], self.goal_pos[0] - self.agent_pos[0]))

        return self.grid.copy()/5, self.reward, done, {}

    def render(self):
        # print(self.grid)
        visualize_grid(self.grid, self.agent_pos, self.agent_orientation)
        
if __name__ == "__main__":
    env = GridEnvironment()
    env.reset()
    env.render()
    
    # obstacle = -30
    # if akinito -> reward = -15
    # if kinithike or sosti gonia -> reward = 0
    # if simeio = simeio termatismou -> done = True
    