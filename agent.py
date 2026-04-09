import numpy as np

class Agent:
    def __init__(self, params):
        if len(params) != 4:
            raise Exception("Invalid argument to agent construtor! Must be a list of 4 parameters.")
        self.params = params

    def chooseAction(self, observation, verbose=False):

        p = self.rightProbability(observation)
        sample = np.random.uniform(0,1)

        if p < sample:
            action = 1
        else:
            action = 0

        if verbose:
            print("score:", score)
            print("p:", p)
            print("sample:", sample)
            print("action:", action)
            print()
        return action

    def rightProbability(self, observation):
        w,x,y,z = observation
        a,b,c,d = self.params

        score = a*w + b*x + c*y + d*z
        p = 1/(1+np.exp(-1*score))
        return p

    def updateWeights(self, nudge):
        self.params = [num + param for num, param in zip(nudge, self.params)]

    def __repr__(self):
        return str([self.a, self.b, self.c, self.d])