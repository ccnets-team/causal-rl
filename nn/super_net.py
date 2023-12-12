import torch
import torch.nn as nn
import torch.nn.functional as F

class SuperNetBlock(nn.Module):
    def __init__(self, input_size, output_size):
        super(SuperNetBlock, self).__init__()
        # Different possible operations within the block
        self.linear1 = nn.Linear(input_size, output_size)
        self.linear2 = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()
        
        # Architecture weights for each operation
        self.arch_weights = nn.Parameter(torch.ones(2) / 2)  # Initialize close to a uniform distribution
        
    def forward(self, x):
        weights = F.softmax(self.arch_weights, dim=-1)
        out = weights[0] * self.linear1(x) + weights[1] * self.linear2(x)
        out = self.relu(out)
        return out

class SuperNet(nn.Module):
    def __init__(self, num_layer, hidden_size, dropout = 0.0):
        super(SuperNet, self).__init__()
        self.num_layer = num_layer

        layers = []
        for i in range(self.num_layer):
            layers.append(SuperNetBlock(hidden_size, hidden_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x, mask = None):
        return self.net(x)