import torch, pygame, random
import numpy as np
from torch import nn
import torch.optim as optim

# Initialize Pygame
pygame.init()

# Set up some constants
WIDTH, HEIGHT = 600, 600
ROWS, COLS = 10, 10
SQUARE_SIZE = WIDTH // COLS

# Set up the display
WIN = pygame.display.set_mode((WIDTH, HEIGHT))

# Set up the title and icon
pygame.display.set_caption("Pac-man AI")
# icon = pygame.image.load("assets/pacman.png")
# pygame.display.set_icon(icon)

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 200, 0)
RED = (200, 0, 0)

# Grid class
class Grid:
	def __init__(self):
		self.grid = np.zeros((ROWS, COLS))
		self.reset()

	def draw(self, win):
		# Draw grid lines
		for i in range(ROWS):
			for j in range(COLS):
				pygame.draw.rect(win, WHITE, (j*SQUARE_SIZE, i*SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 1)		
		
		# Draw pac-dots

		# Draw Pac-man

		# Draw ghosts


	def reset(self):
		# Reset the game
		pass



grid = Grid()
WIN.fill((0, 0, 0))
grid.draw(WIN)
pygame.display.update()

# Main loop
run = True
while run:
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			run = False
	
pygame.quit()