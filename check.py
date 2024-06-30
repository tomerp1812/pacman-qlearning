import pacman
import random
import ex2
import time

def evaluate(board, steps, i, p = 0.7):    
    """Run solver function on given problem and evaluate it's effectiveness."""
    run_pacman = pacman.Game(steps, board) 
    controller = ex2.Controller(len(board), len(board[0]), run_pacman.init_locations.copy(), 
                        run_pacman.init_pellets.copy(), steps)
    
    score =  run_pacman.evaluate_policy(controller, p, 30, visualize=False)
    print("the score at round", i,": ",score)
    return score

def main():
    """Print student id and run evaluation on a given game"""
    
    game0 = ((20,10,10,10,10,11),
             (10,10,10,10,41,11),
             (10,11,10,10,11,11),
             (10,11,10,10,10,10),
             (70,10,10,10,11,10))

    avg = 0
    for i in range(5):
        avg += evaluate(game0, 100, i+1, 0.7)
    print("the avarage score is:", avg/5)


if __name__ == '__main__':
    main()
    

