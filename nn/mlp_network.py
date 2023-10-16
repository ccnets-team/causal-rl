from torch import nn
import torch.nn.functional as F

     
class MLP(nn.Module):
    def create_deep_modules(self, layers_size):
        deep_modules = []
        for in_size, out_size in zip(layers_size[:-1], layers_size[1:]):
            deep_modules.append(nn.Linear(in_size, out_size))
            deep_modules.append(nn.ReLU())
        return nn.Sequential(*deep_modules)

    def __init__(self, num_layer, input_size, output_size, hidden_size):
        super(MLP, self).__init__()   

        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())        
        self.deep = self.create_deep_modules([hidden_size] * (num_layer + 1))  # Fixed the argument to create_deep_modules
        layers.append(self.deep)  # Added the deep network to layers
        layers.append(nn.Linear(hidden_size, output_size))  # Final layer to map to output_size
        
        self.net = nn.Sequential(*layers)  # Encapsulate all layers within a Sequential module

    def forward(self, x):
        return self.net(x)  # Pass input through the entire network

    