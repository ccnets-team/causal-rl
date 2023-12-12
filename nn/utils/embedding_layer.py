'''
COPYRIGHT (c) 2022. CCNets, Inc. All Rights reserved.
'''
import torch
from torch import nn
        
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
    
class FactorizedFeatureEmbeddingLayer(nn.Module):
    def __init__(self, num_features, embedding_size, act_fn='tanh'):
        super(FactorizedFeatureEmbeddingLayer, self).__init__()
        self.feature_embeddings = nn.Parameter(torch.randn(num_features, embedding_size))
        self.bias = nn.Parameter(torch.zeros(1, embedding_size))  # Shared bias across features
        self.act_fn = act_fn

    def forward(self, features):
        # Input features shape: [B, S, F]
        
        # Compute linear part
        features_expanded = features.unsqueeze(-1)  # [B, S, F, 1]
        feature_emb_mul = features_expanded * self.feature_embeddings  # [B, S, F, embedding_size]
        linear_part = feature_emb_mul.sum(dim=2) + self.bias  # [B, S, embedding_size]

        # Compute FM part
        # Calculate pairwise interactions between feature embeddings
        squared_feature_emb_sum = torch.sum(feature_emb_mul, dim=2)**2  # [B, S, embedding_size]
        feature_emb_squared_sum = torch.sum(feature_emb_mul**2, dim=2)  # [B, S, embedding_size]
        fm_part = 0.5 * (squared_feature_emb_sum - feature_emb_squared_sum)  # [B, S, embedding_size]

        # Combine linear and FM parts
        output = linear_part + fm_part

        # Apply activation function
        if self.act_fn == "tanh":
            sequence_embeddings = torch.tanh(output)
        elif self.act_fn == "relu":
            sequence_embeddings = torch.relu(output)
        else:
            sequence_embeddings = output

        return sequence_embeddings