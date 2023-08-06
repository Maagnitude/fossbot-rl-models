import pygame
from visualization import win_size
import imageio
import numpy as np
from sb3_contrib import TRPO
from grid_env import GridEnvironment

env = GridEnvironment()
model = TRPO.load("models/TRPO/trpo_gridworld_15x15", env=env)

images = []
obs = model.env.reset()

pygame.init()
screen = pygame.display.set_mode((win_size[0], win_size[1]))
env.render()

for i in range(25):
    reward = 0
    
    # Capture the pygame screen as an image
    img = pygame.surfarray.array3d(screen)
    
    img = np.flipud(np.rot90(img, k=1))
    
    images.append(img)
    
    action, _ = model.predict(obs)
    obs, reward, _, _ = env.step(action)
    
    if reward == 1000.0:
        env.render()
        img = pygame.surfarray.array3d(screen)
    
        img = np.flipud(np.rot90(img, k=1))
        images.append(img)
        break
    else:
        env.render()

imageio.mimsave("gifs/trpo_gridworld_15x15.gif", [np.array(img) for i, img in enumerate(images)], duration=90, loop=0)