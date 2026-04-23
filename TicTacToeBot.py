import torch
import torch.nn as nn

class TicTacToeBot(nn.Module):
    def __init__(self):
        super().__init__()
        # shared trunk
        self.linear1 = nn.Linear(18, 20, bias=True)
        self.linear2 = nn.Linear(20, 20, bias=True)

        # value head
        self.value_head = nn.Linear(20, 1, bias=True)

        # poilcy head
        self.policy_head = nn.Linear(20, 9, bias=True)
    
    def forward(self, x):

        x = x.view(18)

        
        x = self.linear1(x)
        x = torch.relu(x)
        
        x = self.linear2(x)
        x = torch.relu(x)

        value = self.value_head(x)
        value = torch.tanh(value)

        policy = self.policy_head(x)

        return value, policy
        
        