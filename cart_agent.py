import numpy as np

class CartAgent:
    def __init__(self, params):
        if len(params) != 6:
            raise Exception("Invalid argument to agent construtor! Must be a list of 6 parameters.")
        self.params = params

    def chooseAction(self, observation, verbose=False):

        probabilities = self.calcActionProbabilities(observation)

        sample = np.random.uniform(0,1)
        if sample < probabilities[0]:
            action = 0
        elif sample < probabilities[0] + probabilities[1]:
            action = 1
        else:
            action = 2

        if verbose:
            print("probabilities:", probabilities)
            print("sample:", sample)
            print("action:", action)
            print()
        return action

    def calcActionProbabilities(self, observation):
        x,y = observation
        a1, b1, a2, b2, a3, b3 = self.params

        score1 = a1*x + b1*y
        score2 = a2*x + b2*y
        score3 = a3*x + b3*y
        
        totalExp = np.exp(score1) + np.exp(score2) + np.exp(score3)
        return [np.exp(score1)/totalExp, np.exp(score2)/totalExp, np.exp(score3)/totalExp]

    def updateWeights(self, nudge):
        self.params = [num + param for num, param in zip(nudge, self.params)]

    def __repr__(self):
        return str(self.params)