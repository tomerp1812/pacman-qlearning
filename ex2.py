import time
import random
import pacman

"""
in the cube Q_table, the 4 options of UP, DOWN, LEFT, RIGHT defined as follows:
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
"""


class Controller:
    "This class is a controller for a Pacman game."

    def __init__(self, N, M, init_locations, init_pellets, steps):
        self.init_time = time.time()
        """Initialize controller for given game board and number of steps.
        This method MUST terminate within the specified timeout.
        N - board size along the coordinate x
        M - board size along the coordinate y
        init_locations - the locations of ghosts and Pacman in the initial state
        init_locations - the locations of pellets in the initial state
        steps - number of steps the controller will perform
        """
        self.total_steps = 0
        self.init_locations = init_locations
        self.init_pellets = init_pellets
        self.previous_locations = None
        self.previous_pellets = None
        self.previous_step = None
        self.n = N
        self.m = M
        self.init_steps = steps
        self.alpha = 0.05
        self.gamma = 0.9
        self.p = 0.75
        self.p_total_steps = 0
        # key is number of ghost, and value is tuple of number of checks, probability
        self.q_probabilities = {2: (0, 0.5), 3: (0, 0.5), 4: (0, 0.5), 5: (0, 0.5)}
        self.actions = {(-1, 0): "U", (1, 0): "D", (0, -1): "L", (0, 1): "R"}
        # cube not a matrix
        self.Q_Table = [[[0 for _ in range(4)] for _ in range(self.m)] for _ in range(self.n)]
        self.game = pacman.Game(steps, self.create_board(init_locations, init_pellets))
        self.epsilon = 1
        while time.time() - self.init_time < 4.95 and self.epsilon > 0.05:
            self.training()
            self.epsilon *= 0.9999

    # Constructs the game board based on initial player and pellet locations.
    # Each cell in the grid is assigned a value representing ghost, pacman, pellet, or empty space.
    # Parameters:
    def create_board(self, init_locations, init_pellets):
        board = tuple(tuple(
            21 if (i, j) in init_pellets and init_locations[2] == (i, j) else
            31 if (i, j) in init_pellets and init_locations[3] == (i, j) else
            41 if (i, j) in init_pellets and init_locations[4] == (i, j) else
            51 if (i, j) in init_pellets and init_locations[5] == (i, j) else
            20 if init_locations[2] == (i, j) else
            30 if init_locations[3] == (i, j) else
            40 if init_locations[4] == (i, j) else
            50 if init_locations[5] == (i, j) else
            11 if (i, j) in init_pellets else
            70 if (i, j) == init_locations[7] else
            10
            for j in range(self.m)
        )
                      for i in range(self.n)
                      )
        return board

    # Converts action from a string representation to a numerical value.
    def find_action_as_number(self, action):
        act = self.actions[action]
        if act == "U":
            return 0
        if act == "D":
            return 1
        if act == "L":
            return 2
        else:
            return 3

    # Converts action from a numerical value to a tuple representing movement direction.
    def find_action_as_tuple(self, action):
        if action == 0:
            return -1, 0
        if action == 1:
            return 1, 0
        if action == 2:
            return 0, -1
        else:
            return 0, 1

    # Chooses a random action different from the given action number.
    def choose_different_action(self, act_num):
        random_number = act_num
        while random_number == act_num:
            random_number = random.randint(0, 3)
        return self.find_action_as_tuple(random_number)

    # Checks if the specified action will lead to collision with a ghost.
    def check_for_ghost(self, pac, action):
        new_location = (pac[0] + action[0], pac[1] + action[1])
        if new_location == self.game.locations[2]:
            return 2
        elif new_location == self.game.locations[3]:
            return 3
        elif new_location == self.game.locations[4]:
            return 4
        elif new_location == self.game.locations[5]:
            return 5
        return -1

    # Updates the Q-values probabilities for a specific ghost based on the outcome of an event.
    def update_q_probabilities(self, what_ghost, outcome):
        steps, probability = self.q_probabilities[what_ghost]
        steps += 1
        probability = (probability * (steps - 1) + outcome) / steps
        self.q_probabilities[what_ghost] = (steps, probability)

    # Executes the training process for the agent.
    # The agent interacts with the environment for a certain number of steps, updating its Q-values and probabilities.
    def training(self):
        # Reset the game environment
        self.game.reset()

        # Perform the training loop for a specified number of initial steps
        for i in range(self.init_steps):
            # Get the current location of the pac-man
            pac = self.game.locations[7]

            # Explore or exploit actions based on epsilon-greedy policy
            choose = random.random()
            if choose < self.epsilon:
                act_num = self.exploration(pac)
            else:
                act_num = self.exploitation(pac)

            # Convert action number to a tuple representing movement direction
            action = self.find_action_as_tuple(act_num)

            # Introduce randomness by choosing a different action with probability (1 - p)
            random_action = random.random()
            if random_action >= self.p:
                action = self.choose_different_action(act_num)

            # Check for collision with a ghost and update the game board
            what_ghost = self.check_for_ghost(pac, action)
            reward = self.game.update_board(action)

            # Update Q-table based on the action taken and resulting reward
            self.update_q_table(pac, self.game.locations[7], act_num, reward)

            # If the game is over, update Q-values probabilities for ghosts based on the outcome
            if self.game.done:
                if reward < 0:
                    self.update_q_probabilities(what_ghost, 1)
                elif what_ghost != -1:
                    self.update_q_probabilities(what_ghost, 0)
                return
            # If collision with a ghost occurred during the step, update Q-values probabilities accordingly
            elif what_ghost != -1:
                self.update_q_probabilities(what_ghost, 0)

    # Updates the Q-table based on the current state, next state, action taken, and received reward.
    def update_q_table(self, s1, s2, action, reward):
        current_q_value = self.Q_Table[s1[0]][s1[1]][action]
        max_future_q_value = max(self.Q_Table[s2[0]][s2[1]])
        # Q-value update
        self.Q_Table[s1[0]][s1[1]][action] = current_q_value + self.alpha * (
                reward + self.gamma * max_future_q_value - current_q_value)

    # Chooses an action for exploration based on a random selection.
    def exploration(self, s1):
        action = random.randint(0, len(self.Q_Table[s1[0]][s1[1]]) - 1)
        return action

    # Chooses an action for exploitation based on the Q-values in the Q-table.
    def exploitation(self, s1):
        options = self.Q_Table[s1[0]][s1[1]]
        max_action = 0
        max_value = options[0]
        # Iterate through the available actions to find the one with the highest Q-value
        for i in range(1, len(options)):
            if options[i] > max_value:
                max_action = i
                max_value = options[i]
            # If multiple actions have the same Q-value, randomly select one with a probability of 50%
            elif options[i] == max_value:
                if random.random() > 0.5:
                    max_action = i
                    max_value = options[i]
        return max_action

    # Determines the reward for taking a specific action based on the next state, pellet locations, player locations,
    # and probability.
    def find_reward_for_action(self, next_state, pellets, locations, probability):
        reward = 0

        # If the next state is a pellet location
        if next_state in pellets:
            # If there is only one pellet left
            if len(pellets) == 1:
                # Calculate the reward based on the probability of encountering the pellet
                # and the Q-value probability of each player associated with the pellet
                for key, value in locations.items():
                    if value == next_state:
                        reward = probability * (11 * (1 - self.q_probabilities[key][1]) +
                                                (self.q_probabilities[key][1] * -10))
                # If reward remains zero, set it to the default pellet reward
                if reward == 0:
                    reward = probability * 11
            # If there are multiple pellets left
            else:
                # Calculate the reward for encountering the pellet based on each player's Q-value probability
                for key, value in locations.items():
                    if value == next_state:
                        reward += probability * (1 * (1 - self.q_probabilities[key][1]) +
                                                 (self.q_probabilities[key][1] * -10))
                # If reward remains zero, set it to the default pellet reward
                if reward == 0:
                    reward = probability * 1
        # If the next state is not a pellet location
        else:
            # Calculate the penalty for encountering an empty cell based on each player's Q-value probability
            for key, value in locations.items():
                if value == next_state:
                    reward = probability * (self.q_probabilities[key][1] * -10)

        return reward

    # Checks the total reward for taking a specific action based on the current player locations and
    # pellet distribution.
    def check_reward(self, locations, pellets, act_num):
        reward = 0

        # Iterate through all possible actions
        for i in range(len(self.actions)):
            # Calculate the next state based on the current action
            action = self.find_action_as_tuple(i)
            next_state = (locations[7][0] + action[0], locations[7][1] + action[1])

            # Determine the reward based on whether the current action matches the specified action number
            if i == act_num:
                reward += self.find_reward_for_action(next_state, pellets, locations, self.p)
            else:
                reward += self.find_reward_for_action(next_state, pellets, locations, 1 - self.p)

        return reward

    # Updates the probability of repeating the previous move.
    def update_p(self, actual_move):
        outcome = -1
        # Determine the outcome based on whether the actual move matches the previous step
        if actual_move == self.previous_step:
            outcome = 1
        elif actual_move in self.actions:
            outcome = 0

        # Update the probability based on the outcome
        if outcome != -1:
            self.p_total_steps += 1
            self.p = (self.p * (self.p_total_steps - 1) + outcome) / self.p_total_steps

        # Ensure a minimum value for the probability
        if self.p == 0:
            self.p = 0.75

    # Checks if a given move is within the boundaries of the game board.
    def there_is_cell(self, move):
        # Check if the move coordinates are within the boundaries of the game board
        if 0 <= move[0] < self.n and 0 <= move[1] < self.m:
            return True
        return False

    # Chooses the next move for the agent based on current player locations and pellet distribution.
    def choose_next_move(self, locations, pellets):
        # Update the probability of repeating the previous move
        if self.previous_step is not None:
            actual_move = (locations[7][0] - self.previous_locations[7][0],
                           locations[7][1] - self.previous_locations[7][1])
            self.update_p(actual_move)

        # Determine the next move using exploitation
        new_location_pac = locations[7]
        act_num = self.exploitation(new_location_pac)

        # Calculate the reward for the chosen action
        reward = self.check_reward(locations, pellets, act_num)

        # Calculate the next state based on the chosen action
        action = self.find_action_as_tuple(act_num)
        next_state = (locations[7][0] + action[0], locations[7][1] + action[1])

        # If the next state is outside the game board, remain in the current location
        if not self.there_is_cell(next_state):
            next_state = new_location_pac

        # Update the Q-table based on the current and next states, chosen action, and received reward
        self.update_q_table(new_location_pac, next_state, act_num, reward)

        # Update the previous step and locations
        self.previous_step = action
        self.previous_locations = locations

        # Return the chosen action
        return self.actions[action]
