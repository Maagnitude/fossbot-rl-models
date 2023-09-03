import gym
from gym import spaces
import matplotlib.pyplot as plt
import numpy as np
import math
import random


class GridWorldEnv(gym.Env):
    def __init__(self, grid_map, range_gs = False, max_steps=100):
        super(GridWorldEnv, self).__init__()

        self.grid_map = np.array(grid_map)
        self.num_rows, self.num_cols = self.grid_map.shape

        self.agent_position = None
        self.goal_position = None
        self.range_gs = range_gs
        self.action_space = spaces.Discrete(3)  # 3 actions: move_forward, turn_left, turn_right
        
        # Define the observation space as a Box space with shape (9,)
        self.max_distance = np.sqrt(self.num_rows ** 2 + self.num_cols ** 2)  # Maximum possible Euclidean distance
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(7,), dtype=np.float32)

        self.fig = None
        self.ax = None
        self.last_euclidean_distance=1
        self.total_steps = 0
        self.max_steps = max_steps
        self.orientation = 0.0

    def _calculate_observation(self):
        agent_row, agent_col = self.agent_position

        # Calculate Euclidean distance to the goal position
        goal_row, goal_col = self.goal_position
        euclidean_distance = np.linalg.norm([goal_row - agent_row, goal_col - agent_col])
        euclidean_distance /= self.max_distance  # Scale distance to [0, 1]

        # Calculate direction to the goal position
        dx = goal_row - agent_row
        dy = goal_col - agent_col
        goal_direction = math.atan2(dy, dx)
        goal_dir_deg = math.degrees(goal_direction)
        agent_goal_diff = (goal_dir_deg - self.orientation) % 360.0 - 180.0
        agent_goal_diff = (agent_goal_diff - (-180.0)) / (180.0 - (-180.0))
        goal_direction = (goal_direction + math.pi) / (2 * math.pi)  # Normalize to [0, 1]


        # # Calculate obstacles around the agent's position
        # obstacle_distances = []
        # for i in range(-1, 2):
        #     for j in range(-1, 2):
        #         if i == 0 and j == 0:
        #             continue  # Skip the agent's position
        #         row = agent_row + i
        #         col = agent_col + j
        #         if self.grid_map[row, col] == 0:
        #             obstacle_distances.append(0)  # No obstacle
        #         else:
        #             obstacle_distances.append(1)  # Obstacle present
        
        obstacle_distances = self._get_obs_distance(agent_row, agent_col) # a list with 3 values
        
        steps = self.total_steps/self.max_steps
        observation = np.array([euclidean_distance, goal_direction, agent_goal_diff, steps] + obstacle_distances, dtype=np.float32)
        return observation
    
    # def _is_valid_position(self, row, col):
    #     return 0 <= row < self.num_rows and 0 <= col < self.num_cols and self.grid_map[row, col] == 0

    def reset(self, start_position=None, goal_position=None):
        self.last_euclidean_distance=1
        self.total_steps = 0
        if start_position is None:
            self.agent_position = self._random_position()
        else:
            self.agent_position = start_position

        if goal_position is None:
            self.goal_position = self._random_position()
        else:
            self.goal_position = goal_position
        
        if self.range_gs:
            height, width = self.grid_map.shape
            self.agent_position = (5,random.randint(5,width-5))
            self.goal_position = (height-5,random.randint(5,width-5))

        while np.array_equal(self.agent_position, self.goal_position):
            self.goal_position = self._random_position()

        obs = self._calculate_observation()
        return obs
    
    def step(self, action):
        # Convert the action index to a movement direction
        
        action_row, action_col = self._action_to_direction(action)

        # Calculate the new agent position
        new_row = self.agent_position[0] + action_row
        new_col = self.agent_position[1] + action_col



        # Check if the new position is valid (within the grid boundaries)
        #if self._is_valid_position(new_row, new_col):
            # Check if the new position is not blocked by an obstacle
        if self.grid_map[new_row, new_col] == 0:
            self.agent_position = (new_row, new_col)
        else:
            # Penalize collision with obstacle
            #print("Collision")
            reward = -10
            done = True
            obs = self._calculate_observation()  # Update the observation
            return obs, reward, done, {}

        # Determine the reward and whether the episode is done
        done = False
        #reward = 1  # Default reward for each step


        # Flatten the agent position as the observation
        obs = self._calculate_observation()  # Update the observation
        self.last_euclidean_distance = obs[0]

        reward = -obs[0]

        if np.array_equal(self.agent_position, self.goal_position):
            done = True
            #print("Goal Reached")
            reward = 1000  # Reward for reaching the goal

        self.total_steps += 1
        if self.total_steps > self.max_steps:
            done = True
            reward = -10

        return obs, reward, done, {}


    def render(self, mode='human'):
        if self.fig is None or self.ax is None:  # Only create new fig, ax if they don't already exist
            self.fig, self.ax = plt.subplots()

        self.ax.clear()  # Clear existing plot

        # Remainder of your rendering code...
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                if np.array_equal((i, j), self.agent_position):
                    self.ax.add_patch(plt.Rectangle((j, i), 1, 1, color='red'))
                elif np.array_equal((i, j), self.goal_position):
                    self.ax.add_patch(plt.Rectangle((j, i), 1, 1, color='green'))
                elif self.grid_map[i, j] == 1:
                    self.ax.add_patch(plt.Rectangle((j, i), 1, 1, color='black'))
                else:
                    self.ax.add_patch(plt.Rectangle((j, i), 1, 1, color='white', fill=False))

        plt.xlim([0, self.num_cols])
        plt.ylim([0, self.num_rows])
        plt.draw()
        plt.pause(0.0001) 
        plt.show(block=False)  # Add this line to update the window automatically


    def _random_position(self):
        valid_positions = np.argwhere(self.grid_map == 0)
        index = np.random.choice(len(valid_positions))
        return valid_positions[index]

    def _is_valid_position(self, row, col):
        return 0 <= row < self.num_rows and 0 <= col < self.num_cols and self.grid_map[row, col] == 0

    def _action_to_direction(self, action):
        if isinstance(action, np.ndarray):
            action = action.item()
        
        if action == 0:
            if self.orientation == 0.0:
                return (-1, 0)
            elif self.orientation == 45.0:
                return (-1, 1)
            elif self.orientation == 90.0:
                return (0, 1)
            elif self.orientation == 135.0:
                return (1, 1)
            elif self.orientation == 180.0:
                return (1, 0)
            elif self.orientation == 225.0:
                return (1, -1)
            elif self.orientation == 270.0:
                return (0, -1)
            elif self.orientation == 315.0:
                return (-1, -1)     
        elif action == 1:
            self.orientation = (self.orientation + 45.0) % 360.0
            return (0, 0)
        elif action == 2:
            self.orientation = (self.orientation - 45.0) % 360.0
            return (0, 0)
        
        
    def _get_obs_distance(self, agent_row, agent_col):
        sens_dist = []
        for i in range(-45, 50, 45):
            cone_area = (self.orientation + i) % 360
            if cone_area == 0.0:       # NORTH
                cone_x = agent_row - 1
                cone_y = agent_col
            elif cone_area == 45.0:    # NORTH-EAST
                cone_x = agent_row - 1
                cone_y = agent_col + 1
            elif cone_area == 90.0:    # EAST
                cone_x = agent_row
                cone_y = agent_col + 1
            elif cone_area == 135.0:   # SOUTH-EAST
                cone_x = agent_row + 1
                cone_y = agent_col + 1
            elif cone_area == 180.0:   # SOUTH
                cone_x = agent_row + 1
                cone_y = agent_col
            elif cone_area == 225.0:   # SOUTH-WEST
                cone_x = agent_row + 1
                cone_y = agent_col - 1
            elif cone_area == 270.0:   # WEST
                cone_x = agent_row
                cone_y = agent_col - 1
            elif cone_area == 315.0:   # NORTH-WEST
                cone_y = agent_col - 1
                cone_x = agent_row - 1
            
            if self.grid_map[cone_x, cone_y] == 1:
                sens_dist.append(1.0)
            else:
                sens_dist.append(0.0)
        return sens_dist