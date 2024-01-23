import torch

class ExponentialMovingMeanVar:
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
        self.initialized = False

    def get_mean(self):
        return self.mean

    def get_var(self):
        return self.var

    def update(self, x):
        batch_size = x.size(0)
        if batch_size == 0:
            return self  # Do not update if the batch size is zero

        # Convert x to torch.float64 for high precision
        x = x.to(dtype=torch.float64)
        x = x.mean(dim = 0) # 3D -> 2D
        
        if not self.initialized:
            self.mean = x.mean(dim=0)
            self.squared_mean = (x ** 2).mean(dim=0)
            self.var = x.var(dim=0, unbiased=False)
            self.initialized = True
            return
        
        weights = self.alpha * ((1 - self.alpha) ** torch.arange(x.size(0) - 1, -1, -1, device=self.device))[:, None]
        self.mean = torch.sum(weights * x, dim=0) + (1 - torch.sum(weights, dim=0)) * self.mean
        self.squared_mean = torch.sum(weights * (x**2), dim=0) + (1 - torch.sum(weights, dim=0)) * self.squared_mean
        self.var = self.squared_mean - self.mean**2

    def normalize(self, x):
        if len(x) < 1 or not self.initialized:
            return x
        mean = self.get_mean().unsqueeze(0)
        var = self.get_var().unsqueeze(0)
        normalized_x = (x - mean) / (torch.sqrt(var) + 1e-8)
        return normalized_x.to(dtype=x.dtype)

    def save(self, path):
        torch.save({'mean': self.mean, 'squared_mean': self.squared_mean, 'var': self.var, \
            'alpha': self.alpha, 'window_size': self.window_size, 'initialized': self.initialized}, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.mean = checkpoint['mean']
        self.squared_mean = checkpoint['squared_mean']
        self.var = checkpoint['var']
        self.alpha = checkpoint['alpha']
        self.window_size = checkpoint['window_size']
        self.initialized = checkpoint['initialized']
