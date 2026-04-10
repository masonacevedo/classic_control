import numpy as np

import torch
import torch.nn as nn

class CartAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2,3, bias=False)

    def chooseAction(self, observation, verbose=False):
        with torch.no_grad():
            probabilities = self.forward(observation)
        return torch.multinomial(probabilities, 1).item()

    def forward(self, observation):
        x = torch.tensor(observation)
        logits = self.linear(x)
        probabilities = torch.softmax(logits, dim=0)
        return probabilities
        

    def __repr__(self):
        return str(self.params)