import pygame

# Initialize pygame for rendering
pygame.init()

# Set up the window size
win_size = (1100, 1100)
screen = pygame.display.set_mode(win_size)
pygame.display.set_caption("Grid Environment Video")

def visualize_grid(grid, agent_pos, agent_orientation):
    cell_size = win_size[0] // len(grid)
    screen.fill((255, 255, 255))  # Fill the screen with white color
    
    # Load the agent and obstacle icons
    agent_icon = pygame.image.load("images/agent.png")
    obstacle_icon = pygame.image.load("images/obstacle.png")
    goal_icon = pygame.image.load("images/goal.png")
    crushed_icon = pygame.image.load("images/crushed.png")
    finished_icon = pygame.image.load("images/finished.png")
    
    # Resize icons to fit the cell size
    agent_icon = pygame.transform.scale(agent_icon, (cell_size-10, cell_size-10))
    obstacle_icon = pygame.transform.scale(obstacle_icon, (cell_size, cell_size))
    goal_icon = pygame.transform.scale(goal_icon, (cell_size, cell_size))
    crushed_icon = pygame.transform.scale(crushed_icon, (cell_size, cell_size))
    finished_icon = pygame.transform.scale(finished_icon, (cell_size, cell_size))

    for x in range(len(grid)):
        for y in range(len(grid)):
            cell_rect = pygame.Rect(y * cell_size, x * cell_size, cell_size, cell_size)
            
            if grid[x, y] == 0:  # Free block
                color = (255, 255, 255)
                pygame.draw.rect(screen, color, cell_rect)
            elif grid[x, y] == 1:  # Obstacle
                screen.blit(obstacle_icon, cell_rect)
            elif grid[x, y] == 2:  # Agent
                # Rotate agent icon according to its orientation
                rotated_agent_icon = pygame.transform.rotate(agent_icon, -agent_orientation)  
                agent_x = cell_rect.centerx - rotated_agent_icon.get_width() // 2
                agent_y = cell_rect.centery - rotated_agent_icon.get_height() // 2
                # Draw the centered and rotated agent icon on the screen
                screen.blit(rotated_agent_icon, (agent_x, agent_y))
            elif grid[x, y] == 3:  # Goal
                screen.blit(goal_icon, cell_rect)
            elif grid[x, y] == 4:  # Crushed in obstacle
                screen.blit(crushed_icon, cell_rect)
            elif grid[x, y] == 5:  # Finished
                screen.blit(finished_icon, cell_rect)
                
            # Draw grid lines for better visualization
            pygame.draw.rect(screen, (0, 0, 0), cell_rect, 1)

    pygame.display.flip()
    
# pygame.quit()