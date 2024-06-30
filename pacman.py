import random
import time

class Game:    
    """Game class --- presents a Pacman game played for given number of steps."""
    
    def __init__(self, steps, board):
        """Initialize the Game class.       
        steps - represents the number of steps the game is run 
        board - the initial state of the board"""
        self.steps = steps
        self.init = board
        self.set_locations()
        
        self.actions = dict(zip(('L','D','R','U'), ((0,-1), (1,0), (0,1), (-1,0))))
        
        
    def set_locations(self):
        """Sets the locations of ghost and pellets from the initial state.
        Magic numbers for ghosts and Pacman: 
        2 - red, 3 - blue, 4 - yellow, 5 - green and 7 - Pacman."""
        
        self.init_locations = dict.fromkeys((7, 2, 3, 4, 5))
        self.init_pellets = set()
        
        for i, row in enumerate(self.init):
            for j, square in enumerate(row):
                what_is = divmod(square, 10)
                if 2 <= what_is[0] <= 5 or what_is[0] == 7:
                    self.init_locations[what_is[0]] = (i,j)
                if what_is[1] == 1:
                    self.init_pellets.add((i,j))
           
    def reset(self):
        """Resets the board and ghost locations to the initial state."""
        self.done = False
        self.locations = self.init_locations.copy()
        self.pellets = self.init_pellets.copy()
        self.board = list(list(row) for row in self.init)            

    def there_is_cell(self, move):
        if 0 <= move[0] < len(self.init) and 0 <= move[1] < len(self.init[0]):
            return True
        return False

    
    def move_pacman(self, move):    
        move_check = move[0] + self.locations[7][0], move[1] + self.locations[7][1]
        
        if not self.there_is_cell(move_check) or self.board[move_check[0]][move_check[1]] >= 89:
            return 0
        
        what_is = divmod(self.board[move_check[0]][move_check[1]], 10)
        
        if 2 <= what_is[0] <= 5:
            if random.random() < 1 - 0.2 * (what_is[0]):
                self.done = True
                return 0
            else:
                self.locations[what_is[0]] = None

        self.board[self.locations[7][0]][self.locations[7][1]] -= 60
        self.locations[7] = move_check
        self.board[self.locations[7][0]][self.locations[7][1]] = 70
            
        if what_is[1] == 1:
            self.pellets.remove(move_check)
        return what_is[1]
    
    def update_board(self, move):
        """Move the Pacman and return the prize."""
        prize = self.move_pacman(move)
        
        if self.done:
            prize -= 10
        elif not self.pellets:
            prize += 10
            self.done = True
        
        return prize

    def play_game(self, policy, p, visualize = True):
        """Execute given policy, for a given number of steps.
        if Visualize = True, prints all states along execution.
        Returns the reward"""
        reward = 0
        self.done = True
        
        for i in range(self.steps):
            if self.done:
                self.reset()
            
            move = policy.choose_next_move(self.locations.copy(), self.pellets.copy())
            moves = list(self.actions.keys())
            
            if move not in moves:
                print("This is wrong!")
            
            if random.random() < p:
                reward += self.update_board(self.actions[move])
            else:
                moves.remove(move)
                reward += self.update_board(self.actions[random.choice(moves)])
            
            if visualize:
                print(self.board)
                                           
        return reward

    def evaluate_policy(self, policy, p, times, visualize = True):
        return sum([self.play_game(policy, p, visualize) for i in range(times) ]) / (1.0 * times)
