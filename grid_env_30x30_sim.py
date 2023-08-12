import gym
from gym import spaces
import numpy as np
from visualization2 import visualize_grid
import math
from generator import image_to_map
import time

grid = image_to_map('sim_grid.png', 31)
grid = np.array(grid)

class GridEnvironment(gym.Env):
    def __init__(self):
        super(GridEnvironment, self).__init__()
        
        self.grid = grid.copy()
        self.grid_size = grid.shape[0]
        self.free_block_value = 0
        self.obstacle_value = 1
        self.agent_value = 2
        self.goal_value = 3
        self.agent_on_obstacle_value = 4
        self.agent_on_goal_value = 5
        self.goal_pos = (3, 15)
        self.agent_pos = (15, 15)
        self.stand_still = True
        self.agent_orientation = 0.0
        self.goal_from_agent = math.degrees(math.atan2(self.goal_pos[1] - self.agent_pos[1], self.goal_pos[0] - self.agent_pos[0]))
        self.agent_goal_diff = (self.goal_from_agent - self.agent_orientation) % 360.0 - 180.0
        self.hit_obstacle = False
        self.hit_boundary = False
        
        self.actions = ["move_forward", "stop", "rotate_clockwise", "rotate_counterclockwise"] # 45 degree turns and 1 step forward
        self.num_actions = len(self.actions)

        self.action_space = spaces.Discrete(self.num_actions)
        self.max_distance = np.linalg.norm([self.grid_size-2, self.grid_size-2])  # Maximum possible Euclidean distance
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)

        self.last_euclidean_distance = 1
        self.steps = 0
        self.max_steps = 80
        self.last_step = -1
        self.spinning_left = 1
        self.spinning_right = 1

    def reset(self):
        self.grid = grid.copy()
        self.grid_size = grid.shape[0]
        self._stop()
        self._teleport_random()
        while True:
            if self.grid[self.agent_pos] == self.free_block_value:
                break
            else:
                self._teleport_random()       
        self.last_euclidean_distance = 1
        self.agent_orientation = 0.0
        self.goal_pos = (3, 15)
        self.stand_still = True
        self.steps = 0
        self.max_steps = 80
        self.last_step = -1
        self.spinning_left = 1
        self.spinning_right = 1
        self.goal_from_agent = math.degrees(math.atan2(self.goal_pos[1] - self.agent_pos[1], self.goal_pos[0] - self.agent_pos[0]))
        self.hit_obstacle = False
        self.hit_boundary = False
        
        self.grid[self.goal_pos] = self.goal_value
        self.grid[self.agent_pos] = self.agent_value
        
        obs = self._get_state()

        return obs
         

    def step(self, action):
        
        self._make_action(action)
        observation = self._get_state() 
        new_x, new_y = self.agent_pos
        
        reward = 0.0
        done = False
        if self._check_collision():
            # agent collided with obstacle
            reward = -15.0
            done = True
            # print(f"\nObstacle Collision at {self.agent_pos} - Reward: {reward}\n")
            return observation, reward, done, {}

        if self._check_goal():
            # agent reached the goal
            reward = 1000.0
            done = True
            # print(f"\nGoal Reached at {self.agent_pos} - Reward: {reward}\n")
            return observation, reward, done, {}
        
        if abs(self.agent_goal_diff) <= 5.0:
            # agent is facing the goal
            reward = 0.0
        else:
            # agent is not facing the goal
            reward = -1.0 * ((abs(self.agent_goal_diff) - 5.0) / (180.0 - 5.0))
            
        if self.last_step == 2 and action == 2:
            self.spinning_right += 1
            self.spinning_left = 1
        elif self.last_step == 3 and action == 3:
            self.spinning_left += 1
            self.spinning_right = 1
        else:
            self.spinning_left = 1
            self.spinning_right = 1
            
        if self.spinning_left > 15 or self.spinning_right > 15:
            # fossbot is spinning in place
            reward = -15.0
            done = True
            # print(f"\nSpinning in place - Action Reward: {reward}\n")
            return observation, reward, done, {}
        
        self.last_step = action
        
        if self.steps > self.max_steps:
            # max steps reached
            reward = -15.0
            done = True
            # print(f"\nMax Steps Reached - Action Reward: {reward}\n")
            return observation, reward, done, {}

        # print(f"\nAction Reward: {reward}\n")
        
        return observation, reward, done, {}
    
    
    def _make_action(self, action):
        if action == 0:
            self._move_forward()
        elif action == 1:
            self._stop()
        elif action == 2:
            self._rotate_clockwise()
        else:
            self._rotate_counterclockwise()
        self.steps += 1
        # time.sleep(0.25)
        
            
    def _move_forward(self):
        curr_x, curr_y = self.agent_pos
        
        if self.agent_orientation == 0.0:       # NORTH
            curr_x -= 1
        elif self.agent_orientation == 45.0:    # NORTH-EAST
            curr_x -= 1
            curr_y += 1
        elif self.agent_orientation == 90.0:    # EAST
            curr_y += 1
        elif self.agent_orientation == 135.0:   # SOUTH-EAST
            curr_y += 1
            curr_x += 1
        elif self.agent_orientation == 180.0:   # SOUTH
            curr_x += 1
        elif self.agent_orientation == 225.0:   # SOUTH-WEST
            curr_x += 1
            curr_y -= 1
        elif self.agent_orientation == 270.0:   # WEST
            curr_y -= 1
        elif self.agent_orientation == 315.0:   # NORTH-WEST
            curr_y -= 1
            curr_x -= 1
        self.stand_still = False
        
        new_x, new_y = curr_x, curr_y
        
        if self._is_valid_position(new_x, new_y):
            if self.grid[new_x, new_y] != self.obstacle_value:
                # Free previous block
                self.grid[self.agent_pos] = self.free_block_value
                self.agent_pos = (new_x, new_y)
                if self.grid[self.agent_pos] == self.goal_value:
                    # Reached goal
                    self.grid[self.agent_pos] = self.agent_on_goal_value
                else:
                    # Free block
                    self.grid[self.agent_pos] = self.agent_value       
            elif self.grid[new_x, new_y] == self.obstacle_value:
                # HIT OBSTACLE
                self.grid[self.agent_pos] = self.free_block_value
                self.agent_pos = (new_x, new_y)
                self.grid[self.agent_pos] = self.agent_on_obstacle_value
                self.hit_obstacle = True
        else:
            # HIT BOUNDARY
            self.grid[self.agent_pos] = self.free_block_value
            if new_x == self.grid_size - 1:
                self.grid[new_x, new_y] = self.agent_on_obstacle_value
            elif new_x == 0:
                self.grid[new_x, new_y] = self.agent_on_obstacle_value
            elif new_y == self.grid_size - 1:
                self.grid[new_x, new_y] = self.agent_on_obstacle_value
            elif new_y == 0:
                self.grid[new_x, new_y] = self.agent_on_obstacle_value
            self.hit_boundary = True

        
    def _stop(self):
        self.stand_still = True
        
    def _rotate_clockwise(self):
        self.agent_orientation = (self.agent_orientation + 45) % 360
        self.stand_still = True
        
    def _rotate_counterclockwise(self):
        self.agent_orientation = (self.agent_orientation - 45) % 360
        self.stand_still = True    

    def render(self):
        visualize_grid(self.grid, self.agent_pos, self.agent_orientation)
      
    def _is_valid_position(self, row, col):
        return 1 <= row < self.grid_size - 1 and 1 <= col < self.grid_size - 1
      
    def _get_state(self):
        agent_row, agent_col = self._get_position()
        sens_dist = self._get_obs_distance()
        angle = self.agent_orientation
        
        if angle <= 180.0:
            angle = angle
        else:
            angle = angle - 360.0
        
        steps = self.steps
        
        if sens_dist == 999.9:
            sens_dist = self.max_distance
            
        sens_dist_norm = sens_dist / self.max_distance  # Normalize to [0, 1]
        
        dx = self.goal_pos[0] - agent_row
        dy = self.goal_pos[1] - agent_col
        
        euclidean_distance = np.linalg.norm([dx, dy])
        self.last_euclidean_distance = euclidean_distance / self.max_distance   # Normalize to [0, 1]
        
        goal_direction = math.atan2(dy, dx)
        self.goal_from_agent = math.degrees(goal_direction)
        self.agent_goal_diff = (self.goal_from_agent - self.agent_orientation) % 360.0 - 180.0  # agent_goal_diff -> [-180, 180]
        agent_goal_diff = (self.agent_goal_diff - (-180.0)) / (180.0 - (-180.0))        # Normalize to [0, 1]
        
        goal_direction_norm = (goal_direction + math.pi) / (2 * math.pi)  # Normalize to [0, 1]
        angle_norm = (angle - (-180.0)) / (180.0 - (-180.0))        # Normalize to [0, 1]
        steps_norm = steps / self.max_steps     # Normalize to [0, 1]
        
        # Real values
        # print(f"{steps}. Euclidean distance: {euclidean_distance}, Obstacle distance: {sens_dist}, Angle: {self.agent_orientation}\n    Agent-Goal Diff: {self.agent_goal_diff}, Goal Direction: {self.goal_from_agent}")
        # Normalized values
        # print(f"{steps}. Euclidean distance: {self.last_euclidean_distance}, Obstacle distance: {sens_dist_norm}, Angle: {angle_norm}\n    Agent-Goal Diff: {agent_goal_diff}, Goal Direction: {goal_direction_norm}, Steps: {steps_norm}")

        # Calculate obstacles around the agent's position
        # obstacle_distances = []
        # for i in range(-1, 2):
        #     for j in range(-1, 2):
        #         if i == 0 and j == 0:
        #             continue  # Skip the agent's position
        #         row = agent_row + i
        #         col = agent_col + j
        #         if self.grid[row, col] == 0:
        #             obstacle_distances.append(0)  # No obstacle
        #         else:
        #             obstacle_distances.append(1)  # Obstacle present
        # steps = self.steps/self.max_steps
        return np.array([self.last_euclidean_distance, sens_dist_norm, agent_goal_diff, steps_norm])
    
    
    def _check_collision(self):
        if self.hit_obstacle == True or self.hit_boundary == True:
            return True
        else:
            return False
    
    
    def _check_goal(self):
        if self.grid[self.agent_pos] == self.agent_on_goal_value:
            return True
        else:
            return False
    
    def _get_position(self):
        return self.agent_pos
    
    
    def _get_obs_distance(self):
        agent_row, agent_col = self._get_position()
        sens_dist = 999.9
        for i in range(-45, 50, 45):
            cone_area = (self.agent_orientation + i) % 360
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
            
            if self.grid[cone_x, cone_y] == 1:
                sens_dist = 0.2
                break
        return sens_dist
        

    def _teleport_random(self):
        self.agent_pos = (np.random.randint(1, self.grid_size - 2), np.random.randint(1, self.grid_size - 2))   
    
    
if __name__ == "__main__":
    env = GridEnvironment()
    obs = env._get_state()