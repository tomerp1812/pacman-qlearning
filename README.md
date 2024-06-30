# Pacman Game with Q-Learning
This repository contains a Python implementation of the classic Pacman game, enhanced with a Q-learning algorithm to train an agent to play the game effectively. The agent learns to navigate the maze, avoid ghosts, and collect pellets to maximize its score.

## Introduction
This project demonstrates the use of Q-learning, a popular reinforcement learning algorithm, to train an agent to play the Pacman game. The agent learns optimal policies through exploration and exploitation, aiming to achieve the highest possible score.

## Features
- Q-learning algorithm for training the Pacman agent.
- Evaluation of the agent's performance over multiple rounds.
- Calculation and display of average scores and expected high scores.

## Installation

Clone the repository:
```bash
git clone https://github.com/tomerp1812/pacman-qlearning.git
cd pacman-qlearning
```

run:
```bash
python check.py
```

## Implementation Details
### Q-Learning Algorithm
The Q-learning algorithm is implemented in the Controller class located in the ex2.py file. The Q-table is a 3D list that stores Q-values for each state-action pair. The actions are defined as UP, DOWN, LEFT, and RIGHT, corresponding to indices 0, 1, 2, and 3, respectively.

The training process involves the following steps:

- Initialize the Q-table with zeros.
- For each episode, reset the game environment.
- Choose an action based on the epsilon-greedy policy (exploration vs. exploitation).
- Update the Q-values based on the received reward and the maximum future Q-value.
- Adjust the exploration rate (epsilon) to balance exploration and exploitation.

### Pacman Game
The game logic is encapsulated in the Game class located in the pacman.py file. The class handles the initialization of the game board, movement of Pacman and ghosts, and updating the game state based on the agent's actions.

### Evaluation
The check.py script is responsible for evaluating the trained agent. It initializes a game board, runs the training and evaluation process, and prints the scores for each round. The average score is also calculated and displayed at the end.

### Results
After training, the agent is expected to achieve high scores consistently. The script prints the scores for each round and the average score, providing insight into the agent's performance.
