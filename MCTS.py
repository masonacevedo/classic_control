import numpy as np
from TicTacToe import TicTacToe, GameState
import time
import matplotlib.pyplot as plt


C = np.sqrt(2)

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
        # ans += "parent:" + str(self.parent) + "\n"
        ans += "child_nodes:" + str(len(self.child_nodes)) + "\n"
        return ans
        

def runMCTS(t):
    
    initial_state = GameState()
    root_node = initializeTree(initial_state)

    start_time = time.time()
    
    rng = np.random.default_rng()
    # run algorithm for 1 second, just as a starting point.
    # print("root_node")
    # print(root_node)
    while time.time() - start_time < t:
        leaf_node = descendTree(root_node)
        # print("leaf_node before backprop:")
        # print(leaf_node)
        result = rollout(leaf_node, rng)
        performBackpropagation(leaf_node, result)
        # print("leaf_node after backprop: ")
        # print(leaf_node)
        # input("enter to continue")

    
    return root_node


def descendTree(node):

    initial_node = node
    children = initial_node.child_nodes
    current_node = selectChild(children)


    while current_node.child_nodes != [] and not(current_node.game_state.is_over()[0]):
        children = current_node.child_nodes
        current_node = selectChild(children)
    
    # print("found a leaf node")
    # print("current_node:")
    # print(current_node)
    # input("enter to continue")
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
            
            # print("exploit_term:", exploit_term)
            # print("explore_term:", explore_term)
            # print("score:", score)
            # print()

    
    
    maxScoreIndex = scores.index(max(scores))
    return nodes[maxScoreIndex]

def performBackpropagation(node, result):
    game_over, winner = node.game_state.is_over()
    
    
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





avg_rewards = []
t_values = list(range(1,25))
for t in t_values:
    result_node = runMCTS(t)
    avg_reward = result_node.total_reward/result_node.times_visited
    avg_rewards.append(avg_reward)
    print("Monte carlo for", t, "seconds complete.")
    print(t, ":", avg_reward)
    print()

plt.plot(t_values, avg_rewards)
plt.show()

result = runMCTS(1)

childrenTotal = sum([n.times_visited for n in result.child_nodes])
assert(result.times_visited == childrenTotal)

for i in range(0,3):
    for j in range(0,3):
        child0 = result.child_nodes[i]
        childrenOfChild0Total = sum([n.times_visited for n in child0.child_nodes])
        assert(child0.times_visited == childrenOfChild0Total+1)

        grandchild00 = child0.child_nodes[j]
        greatGrandChildrenTotal = sum([n.times_visited for n in grandchild00.child_nodes])

        assert(grandchild00.times_visited == greatGrandChildrenTotal + 1)

# grandchild00 = result.child_nodes[0].child_nodes[0]
# print("grandchild00")
# print(grandchild00)

# for great in grandchild00.child_nodes:
#     print("great grandchild")
#     print(great)