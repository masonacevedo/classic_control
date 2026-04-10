import numpy as np
import torch
import torch.nn as nn

class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(4, 2, bias=False)

    def chooseAction(self, observation):
        with torch.no_grad():
            probabilities = self.forward(observation)
        return torch.multinomial(probabilities,1).item()

    def forward(self, observation):
        x = torch.tensor(observation)
        logits = self.linear_layer(x)
        probs = torch.softmax(logits, dim=0)
        return probs
    
    def __repr__(self):
        return str(self.linear_layer)