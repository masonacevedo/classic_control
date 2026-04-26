import numpy as np
from TicTacToe import TicTacToe, GameState, Board
import time
import matplotlib.pyplot as plt
import pickle
import os
from TicTacToeBot import TicTacToeBot
from collections import deque
from Node import Node

import torch
import torch.nn as nn


MOVES_TO_INDICES = {
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

INDICES_TO_MOVES = {
    0: (0,0),
    1: (0,1),
    2: (0,2),
    3: (1,0),
    4: (1,1),
    5: (1,2),
    6: (2,0),
    7: (2,1),
    8: (2,2)
}


ALL_MOVES = [(row, col) for row in range(0,3) for col in range(0,3)]

C = 2*np.sqrt(2)
        

def runMCTS(agent, num_simulations, *, initial_state = None, root_node = None):

    if root_node and initial_state:
        raise Exception("runMCTS() requires ONE of initial_state or root_node, but BOTH were provided.")
    elif not(root_node) and not(initial_state):
        raise Exception("runMCTS() requires ONE of initial_state or root_node, but NEITHER were provided.")


    if initial_state and not(root_node):
        root_node = initializeTree(agent, initial_state)
    elif not(initial_state) and root_node:
        pass

    for _ in range(0, num_simulations):

        leaf_node = descendTree(agent, root_node)

        is_over, winner = leaf_node.game_state.is_over()

        if is_over:
            result = 1 if winner == leaf_node.game_state.whoseTurn else -1
        else:
            value, probabilities = neural_evaluation_probabilities(agent, leaf_node.game_state)
            result = value

            legal_moves = TicTacToe.get_legal_moves(leaf_node.game_state)
            for move in legal_moves:
                resulting_state = TicTacToe.apply_move(leaf_node.game_state, move)
                learned_p_score = probabilities[MOVES_TO_INDICES[move]]
                new_node = Node(resulting_state, parent = leaf_node, most_recent_move=move, learned_p_score=learned_p_score)
                leaf_node.child_nodes.append(new_node)

        performBackpropagation(leaf_node, result)


    return root_node


def descendTree(agent, node):

    current_node = selectChild(agent, node)
    while current_node.child_nodes != [] and not(current_node.game_state.is_over()[0]):
        current_node = selectChild(agent, current_node)
    
    return current_node
    

def selectChild(agent, parent_node):
    child_nodes = parent_node.child_nodes

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

def neural_evaluation_probabilities(agent, game_state):
    state_as_tensor = state_to_tensor(game_state)

    value, policy_logits = agent.forward(state_as_tensor)

    legal_moves = TicTacToe.get_legal_moves(game_state)
    for move in ALL_MOVES:
        if not(move in legal_moves):
            index_to_mask = MOVES_TO_INDICES[move]
            policy_logits[index_to_mask] = float('-inf')
    
    probabilities = torch.softmax(policy_logits, dim=0)
    return value, probabilities

def neural_evaluation_logits(agent, game_state):

    state_as_tensor = state_to_tensor(game_state)
    value, policy_logits = agent.forward(state_as_tensor)
    return value, policy_logits

def state_to_tensor(game_state):
    tiles = game_state.board.tiles
    opponent = "o" if game_state.whoseTurn == "x" else "x"
    # numpy arrays are faster than python lists!
    tiles = np.array(tiles)

    their_pieces = torch.tensor(tiles == game_state.whoseTurn, dtype = torch.float32)
    my_pieces = torch.tensor(tiles == opponent, dtype = torch.float32)
    ans = torch.stack([my_pieces, their_pieces], dim=0)

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

def initializeTree(agent, initial_state: GameState):

    root_node = Node(initial_state)
    legal_moves = TicTacToe.get_legal_moves(initial_state)

    value, probabilities = neural_evaluation_probabilities(agent, root_node.game_state)

    for move in legal_moves:
        resulting_state = TicTacToe.apply_move(root_node.game_state, move)
        learned_p_score = probabilities[MOVES_TO_INDICES[move]]
        new_node = Node(resulting_state, parent = root_node, most_recent_move=move, learned_p_score=learned_p_score)
        root_node.child_nodes.append(new_node)
    
    return root_node


def self_play(agent, initial_state, num_simulations):

    results = []
    current_node = initializeTree(agent, initial_state)
    current_state = current_node.game_state
    is_over, winner = current_state.is_over()

    rng = np.random.default_rng()
    while not(is_over):
        current_node = runMCTS(agent, num_simulations, root_node = current_node)

        # pretty sure this logic for figuring out the probabilties is more complex than it needs to be
        # but i don't feel like untangling it.
        visit_counts = [(node.most_recent_move, node.times_visited) for node in current_node.child_nodes]
        total_visits = sum([pair[1] for pair in visit_counts])
        probability_tuples = [(pair[0], pair[1]/total_visits) for pair in visit_counts]

        probabilities = [t[-1] for t in probability_tuples]
        
        policy = {node.most_recent_move: p for p, node in zip(probabilities, current_node.child_nodes)}

        formatted_policy = [policy[INDICES_TO_MOVES[i]] if INDICES_TO_MOVES[i] in policy else 0 for i in range(0, 9)]

        new_triplet = [current_node.game_state, formatted_policy, "unknown"]
        results.append(new_triplet)

        sampled_node = rng.choice(current_node.child_nodes, size=1, p=probabilities)[0]

        current_node = sampled_node
        current_state = current_node.game_state
        is_over, winner = current_state.is_over()

    if winner == "x":
        result_number = 1
    elif winner == "o":
        result_number = -1
    else:
        result_number = 0

    for i in range(0, len(results)):
        triplet = results[i]
        triplet[2] = result_number
        result_number *= -1


    return results




def generate_training_data(agent, num_games_self_play, num_simulations):

    t = \
    [
        ["*", "*", "*"],
        ["*", "*", "*"],
        ["*", "*", "*"]
    ]
    whoseTurn = "x"
    custom_board = Board(t)
    initial_state = GameState(whoseTurn, board=custom_board)

    training_data = []
    for _ in range(0, num_games_self_play):
        training_data += self_play(agent, initial_state, num_simulations=num_simulations)

    return training_data


def train(simulations_per_move, games_played_per_batch, loss_factor = 1, num_epochs=20, learning_rate=0.0001, train_steps_per_iteration=100):

    agent = TicTacToeBot()
    optimizer = torch.optim.Adam(agent.parameters(), lr = learning_rate)

    training_buffer = deque(maxlen=10000)
    for epoch in range(0, num_epochs):

        training_buffer += generate_training_data(agent, games_played_per_batch, simulations_per_move)

        for _ in range(0, train_steps_per_iteration):
            optimizer.zero_grad()

            triplet_index = np.random.randint(len(training_buffer))
            triplet = training_buffer[triplet_index]
            state, mcts_policy, outcome = triplet

            mcts_policy = torch.tensor(mcts_policy, dtype = torch.float32)
            
            value, neural_logits = neural_evaluation_logits(agent, state)
            value_loss = (value - outcome) ** 2
            policy_loss = nn.CrossEntropyLoss()(neural_logits, mcts_policy)
            total_loss = (value_loss + policy_loss)
            # print("total_loss:", total_loss)
            total_loss.backward()
            optimizer.step()
        print("epoch:", epoch, " | total_loss:", total_loss)
        
train(
    simulations_per_move=50,
    games_played_per_batch=100,
    num_epochs=200,
    learning_rate=0.001,
    train_steps_per_iteration=200
)