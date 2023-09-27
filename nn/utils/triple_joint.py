'''
COPYRIGHT (c) 2022. CCNets, Inc. All Rights reserved.
'''
import torch
from torch import nn
from .network_init import create_layer

class TripleJointLayer(nn.Module):
    """Base class for creating joint layers for three variables."""
    def __init__(self, joint_type, input_dim1, input_dim2, input_dim3, output_dim):
        super(TripleJointLayer, self).__init__()
        if joint_type == 'cat':
            self.layer_cat = self._initialize_layer(input_dim1 + input_dim2 + input_dim3, output_dim)
        else:
            self.layer_a = self._initialize_layer(input_dim1, output_dim)
            self.layer_b = self._initialize_layer(input_dim2, output_dim)
            self.layer_c = self._initialize_layer(input_dim3, output_dim)
            
    def _initialize_layer(self, input_dim, output_dim):
        return create_layer(input_dim, output_dim, act_fn = "tanh")

    @staticmethod
    def create(inp_size1, inp_size2, inp_size3, outp_size, joint_type):
        joint_map = {
            "none": NoneJointLayer,
            "cat": CatJointLayer,
            "add": AddJointLayer,
        }
        return joint_map[joint_type](joint_type, inp_size1, inp_size2, inp_size3, outp_size)

    def forward(self, tensor_a, tensor_b, tensor_c):
        raise NotImplementedError

class NoneJointLayer(TripleJointLayer):
    def forward(self, tensor_a, tensor_b, tensor_c):
        return self.layer_a(tensor_a), self.layer_b(tensor_b), self.layer_c(tensor_c)

class CatJointLayer(TripleJointLayer):
    """A joint layer that concatenates the outputs of three layers."""
    def forward(self, tensor_a, tensor_b, tensor_c):
        z = torch.cat([tensor_a, tensor_b, tensor_c], dim=-1)
        return self.layer_cat(z)
    
class AddJointLayer(TripleJointLayer):
    """A joint layer that adds the outputs of three linear layers."""
    def forward(self, tensor_a, tensor_b, tensor_c):
        return (self.layer_a(tensor_a) + self.layer_b(tensor_b) + self.layer_c(tensor_c))/3
