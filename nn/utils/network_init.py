from torch import nn
from collections.abc import Iterable

class ModelParams:
    def __init__(self, d_model, num_layers, dropout):
        """
        Initialize a GPT network.	
        Args:
        - d_model (int): Dimension of the model.
        - num_layers (int): Number of layers in the network.
        - dropout (float): Dropout rate.
        """
        self.d_model = d_model
        self.num_layers = num_layers
        self.dropout = dropout
        
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

def init_weights(module):
    """
    Applies Xavier uniform initialization to certain layers in a module and its submodules.
    Args:
        module (nn.Module): The module to initialize.
    """
    if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.MultiheadAttention):
        nn.init.xavier_uniform_(module.in_proj_weight)
        if module.in_proj_bias is not None:
            nn.init.zeros_(module.in_proj_bias)
                    
    for child in module.children():
        init_weights(child)  # Apply recursively to child submodules


def init_log_std(module, init_log_std_value=-2.0):
    """
    Applies Xavier uniform initialization to certain layers in a module and its submodules.
    Args:
        module (nn.Module): The module to initialize.
    """
    if hasattr(module, 'log_std_layer'):
        # Check if log_std_layer is Sequential and contains a Linear layer
        if isinstance(module.log_std_layer, nn.Linear):
            # Initialize the first Linear layer in log_std_layer
            nn.init.zeros_(module.log_std_layer.weight)
            nn.init.constant_(module.log_std_layer.bias, init_log_std_value)
        elif isinstance(module.log_std_layer, nn.Sequential):
            for child in module.log_std_layer.children():
                if isinstance(child, nn.Linear):
                    nn.init.zeros_(child.weight)
                    nn.init.constant_(child.bias, init_log_std_value)    
                    
    for child in module.children():
        init_log_std(child, init_log_std_value)  # Apply recursively to child submodules
        