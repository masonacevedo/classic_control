import numpy as np

import torch
import torch.nn as nn

class CartAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(2,12, bias=True)
        self.linear2 = nn.Linear(12,3, bias=True)

    def chooseAction(self, observation, verbose=False):
        with torch.no_grad():
            probabilities = self.forward(observation)
        return torch.multinomial(probabilities, 1).item()

    def forward(self, observation):
        x = torch.tensor(observation)
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        probabilities = torch.softmax(x, dim=0)
        return probabilities
        

    def __repr__(self):
        return str(self.params)