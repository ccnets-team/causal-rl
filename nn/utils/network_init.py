from torch import nn
from collections.abc import Iterable


# Mapping activation functions to their PyTorch counterparts
ACTIVATION_FUNCTIONS = {
    "softmax": nn.Softmax(dim=-1),
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
    "relu": nn.ReLU()
}

def add_activation_to_layers(layers, activation_function):
    """Appends the specified activation function to the given layers."""
    if activation_function.lower() != "none":
        if activation_function in ACTIVATION_FUNCTIONS:
            layers.append(ACTIVATION_FUNCTIONS[activation_function])
        else:
            raise ValueError(f"Unsupported activation function: {activation_function}")

class RepeatTensor(nn.Module):
    def __init__(self, n_mul):
        super(RepeatTensor, self).__init__()
        self.repeats = [1, n_mul]
        
    def forward(self, x):
        return x.repeat(*self.repeats)
    
def create_layer(input_size = None, output_size = None, act_fn ="none"):
    """Creates a PyTorch layer with optional input and output sizes, and optional activation functions."""
    layers = []
    # add_activation_to_layers(layers, first_act)
    if (input_size is not None) and (output_size is not None):  
        layers.append(nn.Linear(input_size, output_size))
    add_activation_to_layers(layers, act_fn)
    return nn.Sequential(*layers)

def init_weights(param_object):
    if isinstance(param_object, Iterable):
        for layer in param_object:
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.MultiheadAttention):
                nn.init.xavier_uniform_(layer.in_proj_weight)
                if layer.in_proj_bias is not None:
                    nn.init.zeros_(layer.in_proj_bias)
            else:
                # Handle other layer types if needed
                pass
    else:
        if isinstance(param_object, nn.Linear) or isinstance(param_object, nn.Conv2d) or isinstance(param_object, nn.ConvTranspose2d):
            nn.init.xavier_uniform_(param_object.weight)
            if param_object.bias is not None:
                nn.init.zeros_(param_object.bias)
        elif isinstance(param_object, nn.MultiheadAttention):
            nn.init.xavier_uniform_(param_object.in_proj_weight)
            if param_object.in_proj_bias is not None:
                nn.init.zeros_(param_object.in_proj_bias)
        else:
            # Handle other layer types if needed
            pass


