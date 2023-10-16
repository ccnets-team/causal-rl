import torch
from torch import nn
import torch.nn.functional as F

class GatedLinearUnit(nn.Module):
    def __init__(self, units):
        super(GatedLinearUnit, self).__init__()
        self.linear = nn.Linear(units, units)
        self.sigmoid = nn.Linear(units, units)
        
    def forward(self, inputs):
        return self.linear(inputs) * torch.sigmoid(self.sigmoid(inputs))
    
class GatedResidualNetwork(nn.Module):
    def __init__(self, input_units, units, dropout_rate):
        super(GatedResidualNetwork, self).__init__()
        self.relu_dense = nn.Linear(input_units, units)
        self.linear_dense = nn.Linear(units, units)
        self.dropout = nn.Dropout(dropout_rate)
        self.gated_linear_unit = GatedLinearUnit(units)
        self.layer_norm = nn.LayerNorm(units)
        self.project = nn.Linear(input_units, units)
        
    def forward(self, inputs):
        x = F.relu(self.relu_dense(inputs))
        x = self.linear_dense(x)
        x = self.dropout(x)
        if inputs.size(-1) != self.gated_linear_unit.linear.out_features:
            inputs = self.project(inputs)
        x = inputs + self.gated_linear_unit(x)
        x = self.layer_norm(x)
        return x
    
class VariableSelection(nn.Module):
    def __init__(self, num_features, units, dropout_rate):
        super(VariableSelection, self).__init__()
        self.grns = nn.ModuleList([GatedResidualNetwork(1, units, dropout_rate) for _ in range(num_features)])
        self.grn_concat = GatedResidualNetwork(num_features, units, dropout_rate)
        self.softmax = nn.Linear(units, num_features)
        self.num_features = num_features
        
    def forward(self, inputs):
        v = torch.cat(inputs, dim=-1)
        v = self.grn_concat(v)
        v = torch.sigmoid(self.softmax(v)).unsqueeze(-1)

        x = []
        for idx, input_ in enumerate(inputs):
            x.append(self.grns[idx](input_))
        x = torch.stack(x, dim=1)

        outputs = torch.squeeze(torch.matmul(v.transpose(-1, -2), x), dim=1)
        return outputs

class VariableSelectionFlow(nn.Module):
    def __init__(self, num_layer, input_size, output_size, hidden_size = None, dropout_rate = 0.0, dense_units=None):
        super(VariableSelectionFlow, self).__init__()
        self.variableselection = VariableSelection(input_size, output_size, dropout_rate)
        self.dense_units = dense_units
        if dense_units:
            self.dense_list = nn.ModuleList([nn.Linear(output_size, dense_units) for _ in range(input_size)])
        self.num_features = input_size
        
    def forward(self, inputs):
        split_input = torch.split(inputs, 1, dim=-1)
        if self.dense_units:
            l = [self.dense_list[i](split_input[i]) for i in range(self.num_features)]
        else:
            l = list(split_input)
        return self.variableselection(l)

