# Pac-man-AI
Reinforcement Learning applyed to Pac-man

## Inspiration
We've always been fans of classic arcade games and artificial intelligence, so we combined the two and created a Pac-Man using machine learning? We wanted to push the boundaries of how we perceive and play this game by incorporating modern AI techniques.
## What it does
Our version of Pac-Man is a little different. Instead of the player controlling Pac-Man, we've used Q-learning, a form of **reinforcement learning**, to teach Pac-Man to play the game by itself. It learns to navigate the maze and eat the pac-dots. The ghosts, on the other hand, use a backtracking algorithm based on depth-first search (DFS) to relentlessly chase Pac-Man around the maze.
## How we built it
We used Python as our only programming language this time. The game's interface was designed using **Pygame**. The brain of our Pac-Man, the Q-learning algorithm, was implemented using the **Numpy** library. The ghosts' chasing algorithm was designed using a backtracking/DFS method.
## Challenges we ran into
- Implementing the Q-learning algorithm was a significant challenge. We had to fine-tune the learning and exploration parameters to ensure Pac-Man learned effectively.
- Designing the ghosts' chasing algorithm also presented its challenges, as we had to ensure they were a credible threat to Pac-Man while not making them too efficient.
## Accomplishments that we're proud of
- Creating an AI version of Pac-Man, using such an algorithm as Q-learning, that can effectively navigate the maze is an achievement we're proud of.
- We're also thrilled with how retro the interface is.
## What we learned
We learned a great deal about reinforcement learning, particularly Q-learning, and how to implement it in a real-world project. We also gained experience with Pygame and honed our skills in algorithm design and implementation.
## What's next for Pac-man with Machine Learning
We're considering implementing other types of reinforcement learning, such as deep Q-learning, to further improve Pac-Man's performance. We're also interested in improving the ghosts' AI, perhaps by incorporating machine learning for them as well. Additionally, we aim to refine the game's interface and add new features to make it even more engaging. Stay tuned!
