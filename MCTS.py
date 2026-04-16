import numpy as np
from TicTacToe import TicTacToe, GameState, Board
import time
import matplotlib.pyplot as plt
import pickle
import os


C = 2*np.sqrt(2)

# MCTS algorithm: 
    #     Build Tree. 
    #     
    #     Start at root of Tree, descend nodes according to selection rule until
    #     hitting either a leaf node (state who's decisions have not been explored).
    #     If terminal node, done.
    #     If leaf node, then perform rollout. 
    #     Record result at lowest layer of tree, then perform backpropagation upward. 

class Node:
    def __init__(self, game_state, total_reward = 0, times_visited = 0, parent = None, child_nodes=None):
        self.game_state = game_state
        self.total_reward = total_reward
        self.times_visited = times_visited
        self.parent = parent
        self.child_nodes = [child_nodes] if child_nodes is not None else []
    
    def __repr__(self):
        ans = ""
        ans += str(self.game_state) + "\n"
        ans += "total_reward:" + str(self.total_reward) + "\n"
        ans += "times_visited:" + str(self.times_visited) + "\n"
        ans += "child_nodes:" + str(len(self.child_nodes)) + "\n"
        return ans
        

def runMCTS(initial_state, t):
    
    root_node = initializeTree(initial_state)

    start_time = time.time()
    
    rng = np.random.default_rng()
    while time.time() - start_time < t:
        leaf_node = descendTree(root_node)
        result = rollout(leaf_node, rng)
        performBackpropagation(leaf_node, result)

    
    return root_node


def descendTree(node):

    initial_node = node
    children = initial_node.child_nodes
    current_node = selectChild(children)


    while current_node.child_nodes != [] and not(current_node.game_state.is_over()[0]):
        children = current_node.child_nodes
        current_node = selectChild(children)
    
    return current_node
    

def selectChild(nodes):
    scores = []

    for node in nodes:
        if node.times_visited == 0:
            scores.append(float('inf'))
        else:
            n_i = node.times_visited
            n_p = node.parent.times_visited
            exploit_term = node.total_reward/n_i
            
            explore_term = C * np.sqrt(np.log(n_p)/n_i)
            score = exploit_term + explore_term

            scores.append(score)
    
    maxScoreIndex = scores.index(max(scores))
    return nodes[maxScoreIndex]

def performBackpropagation(node, result):
    game_over, winner = node.game_state.is_over()
    result *= -1
    
    
    if not(game_over):
        legal_moves = TicTacToe.get_legal_moves(node.game_state)
        for move in legal_moves:
            resulting_state = TicTacToe.apply_move(node.game_state, move)
            new_node = Node(resulting_state, parent = node)
            node.child_nodes.append(new_node)

    while node.parent:
        node.total_reward += result
        node.times_visited += 1
        result *= -1
        node = node.parent
    
    node.times_visited += 1
    node.total_reward += result

def initializeTree(initial_state: GameState):

    root_node = Node(initial_state)
    legal_moves = TicTacToe.get_legal_moves(initial_state)

    for move in legal_moves:
        resulting_state = TicTacToe.apply_move(root_node.game_state, move)
        new_node = Node(resulting_state, parent = root_node)
        root_node.child_nodes.append(new_node)
    
    return root_node

def rollout(node: Node, rng):

    state = node.game_state

    original_player = state.whoseTurn
    # performs a rollout by playing random moves until the game ends, 
    # then reports result
    
    
    legal_moves = TicTacToe.get_legal_moves(state)
    is_over, winner = TicTacToe.is_over(state)
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

initial_state = GameState()
# custom_tiles = \
# [
#     ["*", "*", "x"],
#     ["*", "*", "*"],
#     ["o", "*", "*"]
# ]
# whoseTurn = "x"
# custom_board = Board(custom_tiles)
# initial_state = GameState(whoseTurn, board=custom_board)


file_name = "result.pkl"
if os.path.exists(file_name):
    with open(file_name, "rb") as f:
        result = pickle.load(f)
else:

    result = runMCTS(initial_state, 30)
    with open(file_name, "wb") as f:
        pickle.dump(result, f)

print("Optimal Tic-Tac-Toe game:")
print("initial state:")
print(initial_state)
print("\n\n")

current_node = result
current_state = current_node.game_state
while current_node.child_nodes:
    current_state = current_node.game_state

    child_scores = [node.times_visited for node in current_node.child_nodes]
    best_child_index = child_scores.index(max(child_scores))
    best_child = current_node.child_nodes[best_child_index]
    current_node = best_child

    print(current_state.board)
    print()
    print()

current_state = current_node.game_state
print(current_state.board)
