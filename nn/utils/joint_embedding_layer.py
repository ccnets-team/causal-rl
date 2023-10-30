'''
COPYRIGHT (c) 2022. CCNets, Inc. All Rights reserved.
'''
import torch
from torch import nn
from ..utils.network_init import create_layer

class JointEmbeddingLayer(nn.Module):
    """
    A layer that applies different types of joint embeddings to its inputs.

    Parameters:
    - joint_type (str): The type of joint operation to apply. Supported types are 'cat' and 'add'.
    - input_dims (list of int): A list of input dimensions, one for each input tensor.
    - output_dim (int): The dimension of the output tensor.
    - act_fn (str): The activation function to apply. Default is 'none'.
    """
    def __init__(self, *input_sizes, output_size, joint_type="cat"):
        super(JointEmbeddingLayer, self).__init__()
        
        if len(input_sizes) == 0:
            raise ValueError("At least one input dimension must be provided")
        
        if not all(dim > 0 for dim in input_sizes):
            raise TypeError("All input dimensions must be positive integers")        

        self.joint_type = joint_type
        self.input_layers = nn.ModuleList()

        if joint_type == 'cat':
            total_input_dim = sum(input_sizes)
            self.output_layer = create_layer(total_input_dim, output_size, act_fn = "tanh")
        elif joint_type == 'add':
            for input_size in input_sizes:
                self.input_layers.append(create_layer(input_size, output_size, act_fn = "tanh"))
    # Make sure to implement the forward method as well

    def forward(self, *inputs):
        """
        Forward pass for the joint embedding layer.

        Parameters:
        - inputs: A variable number of input tensors.

        Returns:
        - The result of applying the joint embedding layer to the inputs.
        """
        if self.joint_type == 'cat':
            x = torch.cat(inputs, dim=-1)
            return self.output_layer(x)
        elif self.joint_type == 'add':
            outputs = [layer(input) for layer, input in zip(self.input_layers, inputs)]
            return torch.sum(torch.stack(outputs, dim=-1), dim=-1)
        elif self.joint_type == 'none':
            return inputs
        else:
            raise ValueError(f"Unsupported joint type: {self.joint_type}")
