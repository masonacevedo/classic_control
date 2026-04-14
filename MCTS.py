import numpy as np
from TicTacToe import TicTacToe, GameState

def runMCTS(state: GameState):
    # MCTS algorithm: 
    #     starting with the current state:
    #        perform a rollout on each option at least once. keep track of 
    #        the number of times each move has been played, along with the value of that move. 
    #        Then, 
    legal_moves = TicTacToe.get_legal_moves(state)

    rollout_data = {}
    
    rng = np.random.default_rng()

    for move in legal_moves:
        resulting_state = TicTacToe.apply_move(state, move)
        rollout_result = rollout(resulting_state, rng)
        visit_count = 1

        state_string = str(resulting_state)
        print("state_string:")
        print(state_string)
        print()
        rollout_data[state_string] = (rollout_result, visit_count)
    
    print("rollout_data")
    print(rollout_data)


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


newGame = GameState()
runMCTS(newGame)

# rng = np.random.default_rng()

# newGame = GameState()
# result = rollout(newGame, rng)
# print("result:", result)