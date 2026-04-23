import numpy as np
from TicTacToe import TicTacToe, GameState, Board
import time
import matplotlib.pyplot as plt
import pickle
import os
from TicTacToeBot import TicTacToeBot
import torch


C = 2*np.sqrt(2)
# Now, for training: 
#    we do a bunch of self-play, and generate a bunch of tuples of the form
#    (state, MCTS policy, outcome)
# Then, we compute the loss as: 
#    1. the difference between the predicted policy from the network and the MCTS policy
#    2. the difference between the predicted value from the network and the outcome! 
#    3. the log of this? maybe? 

# TODO: 
#   stop using rollouts! and start using value predictions from the neural network

# Convention:
# 0 index in output tensor - top left
# 1 index in output tensor - top mid
# 2 index in output tensor - top right
# 3 index in output tensor - mid left
# 4 index in output tensor - mid mid
# 5 index in output tensor - mid right
# 6 index in output tensor - bot left
# 7 index in output tensor - bot mid
# 8 index in output tensor - bot right


class Node:
    def __init__(self, game_state, total_reward = 0, times_visited = 0, parent = None, child_nodes=None, most_recent_move=None, learned_p_score=None):
        self.game_state = game_state
        self.total_reward = total_reward
        self.times_visited = times_visited
        self.parent = parent
        self.child_nodes = [child_nodes] if child_nodes is not None else []
        self.most_recent_move = most_recent_move
        self.learned_p_score = learned_p_score
    
    def __repr__(self):
        ans = ""
        ans += str(self.game_state) + "\n"
        ans += "total_reward:" + str(self.total_reward) + "\n"
        ans += "times_visited:" + str(self.times_visited) + "\n"
        ans += "child_nodes:" + str(len(self.child_nodes)) + "\n"
        ans += "learned_p_score:" + str(self.learned_p_score) + "\n"
        return ans
        

def runMCTS(agent, initial_state, t):
    
    root_node = initializeTree(initial_state)

    start_time = time.time()
    
    rng = np.random.default_rng()
    while time.time() - start_time < t:

        leaf_node = descendTree(agent, root_node)
        # print("leaf node found:")
        # print(leaf_node)
        # input()
        
        is_over, winner = leaf_node.game_state.is_over()

        if is_over:
            result = 1 if winner == leaf_node.game_state.whoseTurn else -1
        else:
            value, probabilities = neural_evaluation(agent, leaf_node.game_state)
            result = value

            # if we hit a leaf node that isn't the end of the game,
            # we create new nodes out of it's children to expand the tree. 
            legal_moves = TicTacToe.get_legal_moves(leaf_node.game_state)
            for move in legal_moves:
                resulting_state = TicTacToe.apply_move(leaf_node.game_state, move)
                learned_p_score = probabilities[moves_to_indices[move]]
                new_node = Node(resulting_state, parent = leaf_node, most_recent_move=move, learned_p_score=learned_p_score)
                leaf_node.child_nodes.append(new_node)
                # print("created node:")
                # print(new_node)
                # input("enter to continue")

        performBackpropagation(leaf_node, result)

    
    return root_node


def descendTree(agent, node):

    initial_node = node
    children = initial_node.child_nodes
    current_node = selectChild(agent, initial_node, children)


    while current_node.child_nodes != [] and not(current_node.game_state.is_over()[0]):
        children = current_node.child_nodes
        current_node = selectChild(agent, current_node, children)
    
    return current_node
    

def selectChild(agent, parent_node, child_nodes):

    scores = []
    for node in child_nodes:
        if node.times_visited == 0:
            scores.append(float('inf'))
        else:
            n_i = node.times_visited
            n_p = node.parent.times_visited
            exploit_term = node.total_reward/n_i
            
            learned_probability = node.learned_p_score
            explore_term = C * learned_probability * np.sqrt(np.log(n_p)/n_i)
            score = exploit_term + explore_term

            scores.append(score)
    
    maxScoreIndex = scores.index(max(scores))
    return child_nodes[maxScoreIndex]

moves_to_indices = {
    (0,0): 0,
    (0,1): 1,
    (0,2): 2,
    (1,0): 3,
    (1,1): 4,
    (1,2): 5,
    (2,0): 6,
    (2,1): 7,
    (2,2): 8
}
all_moves = [(row, col) for row in range(0,3) for col in range(0,3)]
def neural_evaluation(agent, game_state):
    state_as_tensor = state_to_tensor(game_state)

    value, policy_logits = agent.forward(state_as_tensor)
    # print("value:", value)
    # print("policy_logits:", policy_logits)
    # input()
    # print("policy_logits:", policy_logits)
    # print("policy_logits.shape:", policy_logits.shape)
    # print("policy_logits[0]:", policy_logits[0])
    # print("policy_logits[1]:", policy_logits[1])
    # print("policy_logits[2]:", policy_logits[2])
    # input()

    legal_moves = TicTacToe.get_legal_moves(game_state)
    for move in all_moves:
        if not(move in legal_moves):
            index_to_mask = moves_to_indices[move]
            policy_logits[index_to_mask] = float('-inf')
    
    probabilities = torch.softmax(policy_logits, dim=0)
    return value, probabilities

def state_to_tensor(game_state):
    tiles = game_state.board.tiles
    opponent = "o" if game_state.whoseTurn == "x" else "x"
    # numpy arrays are faster than python lists!
    tiles = np.array(tiles)

    their_pieces = torch.tensor(tiles == game_state.whoseTurn, dtype = torch.float32)
    my_pieces = torch.tensor(tiles == opponent, dtype = torch.float32)
    ans = torch.stack([my_pieces, their_pieces], dim=0)
    # print('ans:', ans)
    return ans


def performBackpropagation(node, result):
    game_over, winner = node.game_state.is_over()
    result *= -1

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

    value, probabilities = neural_evaluation(agent, root_node.game_state)

    for move in legal_moves:
        resulting_state = TicTacToe.apply_move(root_node.game_state, move)
        learned_p_score = probabilities[moves_to_indices[move]]
        new_node = Node(resulting_state, parent = root_node, most_recent_move=move, learned_p_score=learned_p_score)
        root_node.child_nodes.append(new_node)
    
    return root_node




custom_tiles = \
[
    ["*", "*", "x"],
    ["*", "*", "o"],
    ["*", "*", "*"]
]
whoseTurn = "x"
custom_board = Board(custom_tiles)
initial_state = GameState(whoseTurn, board=custom_board)
# initial_state = GameState()

agent = TicTacToeBot()
result = runMCTS(agent, initial_state, 30)



current_node = result
current_state = current_node.game_state
while current_node.child_nodes:
    current_state = current_node.game_state

    child_scores = [node.times_visited for node in current_node.child_nodes]
    best_child_index = child_scores.index(max(child_scores))
    best_child = current_node.child_nodes[best_child_index]
    current_node = best_child
    print(current_state.board)

current_state = current_node.game_state
print(current_state.board)
