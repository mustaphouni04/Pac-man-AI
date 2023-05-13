import torch, pygame, random
import numpy as np
from torch import nn
import torch.optim as optim

# Initialize Pygame
pygame.init()

# Set up some constants
WIDTH, HEIGHT = 1000, 400
ROWS, COLS = 10, 25
SQUARE_SIZE = WIDTH // COLS

# Hyperparameters
LEARNING_RATE = 0.01
DISCOUNT_FACTOR = 0.8
PACDOT_REWARD = 1
WALL_PENALTY = -5
MOVE_PENALTY = -0.5
WIN_REWARD = 10
LOSE_PENALTY = -10

# Set up the display
WIN = pygame.display.set_mode((WIDTH, HEIGHT))

# Set up the title and icon
pygame.display.set_caption("Pac-man AI")
icon = pygame.image.load("assets/pacman.png")
pacman_img = pygame.image.load("assets/pacman.png")
pygame.display.set_icon(icon)

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 200, 0)
RED = (200, 0, 0)
YELLOW = (255, 255, 0)

def pos_in_grid(pos):
	# Check if pos is in grid
	return pos[0] >= 0 and pos[0] < ROWS and pos[1] >= 0 and pos[1] < COLS

def state_to_index(state):
    x, y = state
    return x * COLS + y

def state_to_pos(state):
	x, y = state
	return (y * SQUARE_SIZE, x * SQUARE_SIZE)

# Grid class
class Grid:
	def __init__(self):
		self.reset()

	def draw(self, win, agent):
		WIN.fill(BLACK)
		# Draw grid lines
		for i in range(ROWS):
			for j in range(COLS):
				pygame.draw.rect(win, WHITE, (j*SQUARE_SIZE, i*SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 1)		
		
		# Draw pac-dots and walls
		for i in range(ROWS):
			for j in range(COLS):
				if self.matrix[i][j] == 1: #meaning we will draw a dot
					# it has to pass through each cell, so every row and column will be multiplied by the square size and add it to the middle 
					x = j * SQUARE_SIZE + SQUARE_SIZE // 2  # X-coordinate of the dot's center
					y = i * SQUARE_SIZE + SQUARE_SIZE // 2  # Y-coordinate of the dot's center
					radius = 3  # Radius of the dot
					pygame.draw.circle(win, YELLOW, (x, y), radius)
				elif self.matrix[i][j] == 2: #meaning we will draw a wall
							x = j * SQUARE_SIZE + SQUARE_SIZE // 2  # X-coordinate of the top-left corner of the rhomboid
							y = i * SQUARE_SIZE + SQUARE_SIZE // 2  # Y-coordinate of the top-left corner of the rhomboid
							half_width = SQUARE_SIZE // 2  # Half the width of the rhombus
							points = [
								(x, y - half_width),  # Top point
								(x + half_width, y),  # Right point
								(x, y + half_width),  # Bottom point
								(x - half_width, y)  # Left point
							]
							pygame.draw.polygon(WIN, GREEN, points)
		# Draw Pac-man
		WIN.blit(pacman_img, state_to_pos(agent.state))

		# Draw ghosts

		pygame.display.update()

	def reset(self):
		# Reset the game
		self.matrix = np.array([
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 2, 2, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0],
			[0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 1, 2, 0, 0, 0, 0, 0, 0],
			[0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2, 2, 2, 2, 1, 2, 1, 2, 2, 2, 2, 0, 0, 0],
			[0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 0, 0, 0],
			[0, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 2, 1, 2, 2, 2, 2, 2, 2, 2, 1, 2, 0, 0, 0],
			[0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1, 2, 0, 0, 0],
			[0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 0, 0, 0],
			[0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 0, 0, 0],
			[0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]])
	
	def start_state(self):
		# Return the start state
		return (7, 16)
	
	def step(self, state, action):
		# Calculate next state based on action
		# If action leads to a wall, next_state = state
		if action == 0: # Up
			next_state = (state[0] - 1, state[1])
		elif action == 1: # Down
			next_state = (state[0] + 1, state[1])
		elif action == 2: # Left
			next_state = (state[0], state[1] - 1)
		elif action == 3: # Right
			next_state = (state[0], state[1] + 1)

		# Check if game is done
		n_pac_dots = np.sum(self.matrix == 1)
		if n_pac_dots == 0:
			# all pac-dots eaten
			done = True
			reward = WIN_REWARD
		elif pos_in_grid(next_state) and self.matrix[next_state] == -1:
			# Pac-Man caught by ghost
			done = True
			reward = LOSE_PENALTY
		else:
			done = False
			if not pos_in_grid(next_state) or self.matrix[next_state] == 2:
				# out of grid or collision with wall
				next_state = state
				reward = WALL_PENALTY
			elif self.matrix[next_state] == 1:
				# collision with pac-dot
				self.matrix[next_state] = 0
				reward = PACDOT_REWARD
			else:
				# move to empty space
				reward = MOVE_PENALTY

		return next_state, reward, done


# Agent class (Pac-man)
class Agent:
	def __init__(self, num_states, num_actions, alpha=LEARNING_RATE, gamma=DISCOUNT_FACTOR):
		self.num_states = num_states
		self.num_actions = num_actions
		self.alpha = alpha  # learning rate
		self.gamma = gamma  # discount factor
		self.state = (0, 0)

		# Initialize the Q-table to small random values
		self.Q_table = np.random.uniform(low=0, high=1, size=(num_states, num_actions))

	def choose_action(self):
		# Return the action with the highest Q-value for the current state
		# This represents the policy
		index = state_to_index(self.state)
		return np.argmax(self.Q_table[index])

	def update_Q_table(self, action, reward, next_state):
		# Calculate the current Q-value
		index = state_to_index(self.state)
		current_q_value = self.Q_table[index][action]

		# Calculate the maximum Q-value for the next state
		index2 = state_to_index(next_state)
		next_max_q_value = np.max(self.Q_table[index2])

		# Update the Q-value for the current state-action pair
		self.Q_table[index][action] = (1 - self.alpha) * current_q_value + self.alpha * (reward + self.gamma * next_max_q_value)


# Initialize the clock
clock = pygame.time.Clock()
FPS = 3

# Initialize the grid
grid = Grid()
pygame.display.update()

# Initialize the agent
agent = Agent(num_states=ROWS*COLS*ROWS*COLS, num_actions=4)

# Training loop
print("training agent...")
for episode in range(50000): # number of games
	agent.state = grid.start_state()  # Start state from the grid
	done = False
	while not done:
		action = agent.choose_action()
	
		# Assume the game environment provides the next_state and reward.
		next_state, reward, done = grid.step(agent.state, action)
		agent.update_Q_table(action, reward, next_state)
		agent.state = next_state
print("agent trained")

# Main loop
run = True
done = True
while run:
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			run = False

	# Game logic
	# move the agent
	if not done:
		print("agent state: ", agent.state)
		# Choose an action
		action = agent.choose_action()
		print("agent chooses action: ", action)
		# Execute the action and get the new state
		new_state, _, done = grid.step(agent.state, action)
		print("new agent state: ", new_state)
		
		# Update the current state
		agent.state = new_state
	else:
		# Reset the grid and state when game is over
		print("game over, resetting...")
		grid.reset()
		state = grid.start_state()
		done = False

	# Update the display
	grid.draw(WIN, agent)
	clock.tick(FPS)

	
pygame.quit()