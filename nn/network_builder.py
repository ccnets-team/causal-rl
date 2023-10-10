import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.setting.rl_params import NetworkParameters

class NetworkCreator(nn.Module):
    def __init__(self, network_params: NetworkParameters):
        pass
    
    def create(self, num_layer, hidden_size):
        pass