from torch import nn

class ResBlock(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super(ResBlock, self).__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        residual = x
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out += residual
        out = self.relu(out)
        out = self.dropout(out)  
        return out

class ResMLP(nn.Module):
    def __init__(self, num_layer, d_model, dropout = 0.0):
        hidden_dim, num_blocks = d_model, num_layer
        super(ResMLP, self).__init__()
        self.layers = nn.Sequential(
            *(ResBlock(hidden_dim, dropout=dropout) for _ in range(num_blocks))
        )
    
    def forward(self, x, mask = None):
        out = self.layers(x)
        return out
    
class MLP(nn.Module):
    def create_deep_modules(self, layers_size, dropout = 0.0):
        deep_modules = []
        for in_size, out_size in zip(layers_size[:-1], layers_size[1:]):
            deep_modules.append(nn.Linear(in_size, out_size))
            deep_modules.append(nn.ReLU())
            deep_modules.append(nn.Dropout(dropout))
        return nn.Sequential(*deep_modules)

    def __init__(self, num_layer, d_model, dropout = 0.0):
        super(MLP, self).__init__()   
        self.deep = self.create_deep_modules([d_model] + [int(d_model) for i in range(num_layer)], dropout)
                
    def forward(self, x, mask = None):
        x = self.deep(x)
        return x
    