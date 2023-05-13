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
icon = pygame.image.load("assets/pacman.png")
pacman_img = pygame.image.load("assets/pacman.png")
pygame.display.set_icon(icon)

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 200, 0)
RED = (200, 0, 0)

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
		self.matrix = np.ones((ROWS, COLS))
	
	def start_state(self):
		# Return the start state
		return (0, 0)
	
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
			reward = 10
		elif pos_in_grid(next_state) and self.matrix[next_state] == -1:
			# Pac-Man caught by ghost
			done = True
			reward = -10
		else:
			done = False
			if not pos_in_grid(next_state) or self.matrix[next_state] == 2:
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
WIN.fill((0, 0, 0))
grid.draw(WIN)
pygame.display.update()

# Initialize the agent
agent = Agent(num_states=ROWS*COLS*ROWS*COLS, num_actions=4)

# Training loop
print("training agent...")
for episode in range(10): # number of games
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
		
		# Render the new state of the grid
		WIN.fill((0, 0, 0))
		grid.draw(WIN)
		
		# Update the current state
		agent.state = new_state
	else:
		# Reset the grid and state when game is over
		print("game over, resetting...")
		grid.reset()
		state = grid.start_state()
		done = False

	# Update the display
	WIN.blit(pacman_img, state_to_pos(agent.state))
	pygame.display.update()
	clock.tick(FPS)

	
pygame.quit()