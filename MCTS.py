import numpy as np
from TicTacToe import TicTacToe, GameState

def runMCTS(state: GameState):
    # MCTS algorithm: 
    #     starting with the current state:
    #        perform a rollout on each option at least once. keep track of 
    #        the number of times each move has been played, along with the value of that move. 
    #        Then, 
    pass


def rollout(state: GameState, rng):

    original_player = state.whoseTurn
    # performs a rollout by playing random moves until the game ends, 
    # then reports result
    
    
    legal_moves = TicTacToe.get_legal_moves(state)
    is_over = False
    while len(legal_moves) > 0 and not(is_over):
        
        move_to_perform = rng.choice(legal_moves)
        state = TicTacToe.apply_move(state, move_to_perform)
        legal_moves = TicTacToe.get_legal_moves(state)
        is_over, winner = TicTacToe.is_over(state)
    
    if winner == None:
        return 0
    elif winner == original_player:
        return 1
    else:
        return -1

rng = np.random.default_rng()

newGame = GameState()
result = rollout(newGame, rng)
print("result:", result)