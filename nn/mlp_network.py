from torch import nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, hidden_dim):
        super(ResBlock, self).__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out += residual
        out = self.relu(out)
        return out

class ResMLP(nn.Module):
    def __init__(self, num_layer, hidden_size):
        hidden_dim, num_blocks = hidden_size, num_layer
        super(ResMLP, self).__init__()
        self.layers = nn.Sequential(
            *(ResBlock(hidden_dim) for _ in range(num_blocks))
        )

    def forward(self, x, mask = None):
        out = self.layers(x)
        return out
    
class MLP(nn.Module):
    def create_deep_modules(self, layers_size):
        deep_modules = []
        for in_size, out_size in zip(layers_size[:-1], layers_size[1:]):
            deep_modules.append(nn.Linear(in_size, out_size))
            deep_modules.append(nn.ReLU())
        return nn.Sequential(*deep_modules)

    def __init__(self, num_layer, hidden_size):
        super(MLP, self).__init__()   
        self.deep = self.create_deep_modules([hidden_size] + [int(hidden_size) for i in range(num_layer)])
                
    def forward(self, x, mask = None):
        x = self.deep(x)
        return x
    