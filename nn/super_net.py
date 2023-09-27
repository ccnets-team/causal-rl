import torch
import torch.nn as nn
import torch.nn.functional as F

class SuperNetBlock(nn.Module):
    def __init__(self, hidden_dim):
        super(SuperNetBlock, self).__init__()
        # Different possible operations within the block
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        
        # Architecture weights for each operation
        self.arch_weights = nn.Parameter(torch.ones(2) / 2)  # Initialize close to a uniform distribution
    def forward(self, x):
        weights = F.softmax(self.arch_weights, dim=-1)
        out = weights[0] * self.linear1(x) + weights[1] * self.linear2(x)
        out = self.relu(out)
        return out

class SuperNet(nn.Module):
    def __init__(self, num_layer, hidden_dim):
        super(SuperNet, self).__init__()
        self.num_layer = num_layer

        layers = []
        for i in range(self.num_layer):
            layers.append(SuperNetBlock(hidden_dim))
                
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)