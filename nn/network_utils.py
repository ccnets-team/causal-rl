
'''
COPYRIGHT (c) 2022. CCNets, Inc. All Rights reserved.
'''
import torch
from torch import nn

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
        
class ContinuousFeatureEmbeddingLayer(nn.Module):
    def __init__(self, num_features, embedding_size, act_fn='tanh'):
        super(ContinuousFeatureEmbeddingLayer, self).__init__()
        self.feature_embeddings = nn.Parameter(torch.randn(num_features, embedding_size))
        self.bias = nn.Parameter(torch.zeros(1, embedding_size))  # Shared bias across features
        self.act_fn = act_fn

    def forward(self, features):
        # Input features shape: [B, S, F]
        # B: Batch size, S: Sequence length, F: Number of features
        features_expanded = features.unsqueeze(-1)
        # After unsqueeze, features shape: [B, S, F, 1]
        
        # self.feature_embeddings shape: [F, embedding_size]
        # We broadcast multiply features with embeddings to get a shape: [B, S, F, embedding_size]
        feature_emb_mul = features_expanded * self.feature_embeddings
        
        # Sum across the feature dimension F, resulting shape: [B, S, embedding_size]
        feature_emb_bias = feature_emb_mul.sum(dim=2) + self.bias  # Sum first, then add bias

        if self.act_fn == "tanh":
            sequence_embeddings = torch.tanh(feature_emb_bias)
        elif self.act_fn == "relu":
            sequence_embeddings = torch.relu(feature_emb_bias)
        else:
            sequence_embeddings = feature_emb_bias

        return sequence_embeddings