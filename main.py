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

def pos_in_grid(pos):
	# Check if pos is in grid
	return pos[0] >= 0 and pos[0] < ROWS and pos[1] >= 0 and pos[1] < COLS

# Grid class
class Grid:
	def __init__(self):
		self.matrix = np.zeros((ROWS, COLS))
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
	
	def start_state(self):
		# Return the start state
		return (0, 0)
	
	def step(self, state, action):
		# Calculate next state based on action
		# If action leads to a wall, next_state = state
		if action == 0: # Up
			next_state = (state[0] - 1, state[1])
			if self.matrix[next_state] == -1:
				next_state = state
		elif action == 1: # Down
			next_state = (state[0] + 1, state[1])
			if self.matrix[next_state] == -1:
				next_state = state
		elif action == 2: # Left
			next_state = (state[0], state[1] - 1)
			if self.matrix[next_state] == -1:
				next_state = state
		elif action == 3: # Right
			next_state = (state[0], state[1] + 1)
			if self.matrix[next_state] == -1:
				next_state = state

		# Check if game is done
		n_pac_dots = np.sum(self.matrix == 1)
		if n_pac_dots == 0:
			# all pac-dots eaten
			done = True
			reward = 10
		elif pos_in_grid(next_state) and self.matrix[next_state] == -2:
			# Pac-Man caught by ghost
			done = True
			reward = -10
		else:
			done = False
			if not pos_in_grid(next_state) or self.matrix[next_state] == -1:
				# out of grid or collision with wall
				next_state = state
				reward = -1
			elif self.matrix[next_state] == 1:
				# collision with pac-dot
				self.matrix[next_state] = 0
				reward = 1
			else:
				# no collision
				reward = -0.1

		return next_state, reward, done


# Agent class (Pac-man)
class Agent:
	def __init__(self, num_states, num_actions, alpha=0.5, gamma=0.95):
		self.num_states = num_states
		self.num_actions = num_actions
		self.alpha = alpha  # learning rate
		self.gamma = gamma  # discount factor

		# Initialize the Q-table to small random values
		self.Q_table = np.random.uniform(low=0, high=1, size=(num_states, num_actions))

	def choose_action(self, state):
		# Return the action with the highest Q-value for the current state
		# This represents the policy
		return np.argmax(self.Q_table[state])

	def update_Q_table(self, state, action, reward, next_state):
		# Calculate the current Q-value
		current_q_value = self.Q_table[state][action]

		# Calculate the maximum Q-value for the next state
		next_max_q_value = np.max(self.Q_table[next_state])

		# Update the Q-value for the current state-action pair
		self.Q_table[state][action] = (1 - self.alpha) * current_q_value + self.alpha * (reward + self.gamma * next_max_q_value)

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