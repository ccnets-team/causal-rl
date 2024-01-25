import torch

class HybridMovingMeanVar:
    def __init__(self, num_features, device, alpha=None, window_size=None):
        assert (alpha is None or window_size is None), "Only one of alpha or window_size can be specified"
        if window_size is not None:
            alpha = 2 / (window_size + 1)
        assert alpha is not None, "Either alpha or window_size must be specified"
        assert 0 < alpha < 1, "alpha must be between 0 and 1 (exclusive)"
        
        self.alpha = alpha
        self.window_size = window_size if window_size is not None else (2 / alpha - 1)
        self.device = device
        self.mean = torch.zeros(num_features, device=self.device, dtype=torch.float64)
        self.squared_mean = torch.zeros(num_features, device=self.device, dtype=torch.float64)
        self.var = torch.ones(num_features, device=self.device, dtype=torch.float64)
        self.count = 0
        self.initialized = False

    def get_mean(self):
        return self.mean

    def get_var(self):
        return self.var

    def update(self, x, padding_mask=None):
        batch_size, sequence_length, num_features = x.size()
        if batch_size == 0:
            return self  # Do not update if the batch size is zero

        # Convert x to torch.float64 for high precision
        x = x.to(dtype=torch.float64)
        total_elements = int(batch_size*sequence_length)
        new_count = self.count + total_elements

        if not self.initialized:
            # Compute the mean and variance across both batch and sequence dimensions
            self.mean = x.mean(dim=(0, 1))
            self.squared_mean = (x ** 2).mean(dim=(0, 1))
            self.var = x.var(dim=(0, 1), unbiased=False)
            self.count = new_count
            self.initialized = True
            return
        
        # Compute weights for each time step and expand them to match x's shape
        exponential_weights = self.alpha * ((1 - self.alpha) ** torch.arange(sequence_length-1, -1, -1, device=self.device))
        exponential_weights = exponential_weights.view(1, sequence_length, 1).expand_as(x)/total_elements

        # Compute an adjustment factor based on batch size and total count
        hybrid_weights = exponential_weights + (1 / new_count)
        
        sum_hybrid_weights = torch.sum(hybrid_weights, dim=(0, 1)).expand_as(hybrid_weights)
        hybrid_weights = torch.where(sum_hybrid_weights > 1, hybrid_weights/sum_hybrid_weights, hybrid_weights)

        # Apply padding mask
        if padding_mask is not None:
            weights = hybrid_weights * padding_mask.to(dtype=x.dtype)
            missing_elements = total_elements - padding_mask.sum()
            new_count -= missing_elements
        else:
            weights = hybrid_weights


        self.mean = torch.sum(weights * x, dim=(0, 1)) + (1 - torch.sum(weights, dim=(0, 1))) * self.mean
        self.squared_mean = torch.sum(weights * (x**2), dim=(0, 1)) + (1 - torch.sum(weights, dim=(0, 1))) * self.squared_mean
        self.var = self.squared_mean - self.mean**2
        self.count = new_count

    def normalize(self, x):
        if len(x) < 1 or not self.initialized:
            return x
        mean = self.get_mean().view(1, 1, -1)
        var = self.get_var().view(1, 1, -1)
        normalized_x = (x - mean) / (torch.sqrt(var) + 1e-8)
        return normalized_x.to(dtype=x.dtype)

    def save(self, path):
        torch.save({'mean': self.mean, 'squared_mean': self.squared_mean, 'var': self.var, \
            'count': self.count, 'alpha': self.alpha, 'window_size': self.window_size, 'initialized': self.initialized}, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.mean = checkpoint['mean']
        self.squared_mean = checkpoint['squared_mean']
        self.var = checkpoint['var']
        self.alpha = checkpoint['alpha']
        self.window_size = checkpoint['window_size']
        self.initialized = checkpoint['initialized']
        self.count = checkpoint['count']
