import gym
from gym import spaces
import numpy as np
from visualization import visualize_grid
import math

grid = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
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

class GridEnvironment(gym.Env):
    def __init__(self, max_steps=100):
        super(GridEnvironment, self).__init__()
        
        self.grid = grid.copy()
        self.grid_size = 15
        self.free_block_value = 0
        self.obstacle_value = 1
        self.agent_value = 2
        self.goal_value = 3
        self.agent_on_obstacle_value = 4
        self.agent_on_goal_value = 5
        self.reward = 0
        self.stand_still = True
        self.agent_orientation = 90
        self.goal_from_agent = 45
        
        self.actions = ["TurnLeft", "TurnRight", "MoveForward"] # 45 degree turns and 1 step forward
        self.num_actions = len(self.actions)

        self.action_space = spaces.Discrete(self.num_actions)
        self.max_distance = np.sqrt((self.grid_size-3) ** 2 + (self.grid_size-3) ** 2)  # Maximum possible Euclidean distance
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(12,), dtype=np.float32)

        self.last_euclidean_distance = 1
        self.steps = 0
        self.max_steps = max_steps

    def reset(self):
        self.grid = grid.copy()
        self.last_euclidean_distance=1
        self.agent_pos = (1, 1)
        self.agent_orientation = 90
        self.goal_pos = (self.grid_size - 2, self.grid_size - 2)
        self.reward = 0
        self.stand_still = True
        self.steps = 0
        self.goal_from_agent = math.degrees(math.atan2(self.goal_pos[1] - self.agent_pos[1], self.goal_pos[0] - self.agent_pos[0]))
        
        self.grid[self.goal_pos] = self.goal_value
        self.grid[self.agent_pos] = self.agent_value
        
        obs = self._calculate_observation()

        return obs

    def step(self, action):
        x, y = self.agent_pos
        done = False

        if action == 0:  # TurnLeft
            self.agent_orientation = (self.agent_orientation - 45) % 360
            self.stand_still = True
        elif action == 1:  # TurnRight
            self.agent_orientation = (self.agent_orientation + 45) % 360
            self.stand_still = True
        elif action == 2:  # MoveForward
            if self.agent_orientation == 0:  # NORTH
                x -= 1
            elif self.agent_orientation == 45: # NORTH-EAST
                x -= 1
                y += 1
            elif self.agent_orientation == 90:  # EAST
                y += 1
            elif self.agent_orientation == 135:  # SOUTH-EAST
                x += 1
                y += 1
            elif self.agent_orientation == 180:  # SOUTH
                x += 1
            elif self.agent_orientation == 225:  # SOUTH-WEST
                x += 1
                y -= 1
            elif self.agent_orientation == 270:  # WEST
                y -= 1  
            elif self.agent_orientation == 315:  # NORTH-WEST
                x -= 1
                y -= 1 
            self.stand_still = False  

        # Check if the new position is valid
        if self._is_valid_position(x, y):
            if self.grid[x, y] != self.obstacle_value and self.stand_still == False:
                self.grid[self.agent_pos] = self.free_block_value
                self.reward = 0.0
                self.agent_pos = (x, y)
                if self.grid[self.agent_pos] == self.goal_value:
                    self.reward = 1000.0
                    self.grid[self.agent_pos] = self.agent_on_goal_value
                    done = True
                    obs = self._calculate_observation()
                    return obs, self.reward, done, {}
                else:
                    self.grid[self.agent_pos] = self.agent_value
                    obs = self._calculate_observation()        
            elif self.stand_still == True:
                obs = self._calculate_observation()
                if self.agent_goal_diff > 5:
                    self.reward = -1.0
                else:
                    self.reward = 0.0
            elif self.grid[x, y] == self.obstacle_value:
                # HIT OBSTACLE
                self.grid[self.agent_pos] = self.free_block_value
                self.reward = -10.0
                self.agent_pos = (x, y)
                self.grid[self.agent_pos] = self.agent_on_obstacle_value
                done = True
                obs = self._calculate_observation()
                return obs, self.reward, done, {}
        else:
            # HIT BOUNDARY
            self.grid[self.agent_pos] = self.free_block_value
            if x == self.grid_size - 1:
                self.grid[x, y] = self.agent_on_obstacle_value
            elif x == 0:
                self.grid[x, y] = self.agent_on_obstacle_value
            elif y == self.grid_size - 1:
                self.grid[x, y] = self.agent_on_obstacle_value
            elif y == 0:
                self.grid[x, y] = self.agent_on_obstacle_value
            self.reward = -15.0
            done = True
            obs = self._calculate_observation()
            return obs, self.reward, done, {}

        self.goal_from_agent = math.degrees(math.atan2(self.goal_pos[1] - self.agent_pos[1], self.goal_pos[0] - self.agent_pos[0]))

        # Flatten the agent position as the observation
        self.last_euclidean_distance = obs[0]
        
        self.reward = -obs[0]
        
        self.steps += 1
        if self.steps > self.max_steps:
            done = True
            self.reward = -10.0
            
        # return self.grid.copy()/5, self.reward, done, {}
        return obs, self.reward, done, {}

    def render(self):
        visualize_grid(self.grid, self.agent_pos, self.agent_orientation)
      
    def _is_valid_position(self, row, col):
        return 1 <= row < self.grid_size - 1 and 1 <= col < self.grid_size - 1
      
    def _calculate_observation(self):
        agent_row, agent_col = self.agent_pos

        # Calculate Euclidean distance to the goal position
        goal_row, goal_col = self.goal_pos
        euclidean_distance = np.linalg.norm([goal_row - agent_row, goal_col - agent_col])
        euclidean_distance /= self.max_distance  # Scale distance to [0, 1]

        # Calculate direction to the goal position
        dx = goal_row - agent_row
        dy = goal_col - agent_col
        goal_direction = math.atan2(dy, dx)
        self.goal_from_agent = math.degrees(goal_direction)
        self.agent_goal_diff = abs(self.agent_orientation - self.goal_from_agent - 90)
        agent_goal_diff = self.agent_goal_diff/360 # Normalize to [0, 1]
        
        goal_direction = (goal_direction + math.pi) / (2 * math.pi)  # Normalize to [0, 1]

        # Calculate obstacles around the agent's position
        obstacle_distances = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue  # Skip the agent's position
                row = agent_row + i
                col = agent_col + j
                if self.grid[row, col] == 0:
                    obstacle_distances.append(0)  # No obstacle
                else:
                    obstacle_distances.append(1)  # Obstacle present
        steps = self.steps/self.max_steps
        observation = np.array([euclidean_distance, goal_direction, agent_goal_diff, steps] + obstacle_distances, dtype=np.float32)
        return observation
    