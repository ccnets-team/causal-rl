import torch

class ExponentialMovingAbsMean:
    def __init__(self, num_features, device, alpha=None, window_size=None):
        assert (alpha is None or window_size is None), "Only one of alpha or window_size can be specified"
        if window_size is not None:
            alpha = 2 / (window_size + 1)
        assert alpha is not None, "Either alpha or window_size must be specified"
        assert 0 < alpha < 1, "alpha must be between 0 and 1 (exclusive)"
        
        self.alpha = alpha
        self.window_size = window_size if window_size is not None else (2 / alpha - 1)
        self.device = device
        self.abs_mean = torch.zeros(num_features, device=self.device, dtype=torch.float64)
        self.initialized = False

    def update(self, x):
        batch_size = x.size(0)
        if batch_size == 0:
            return self  # Do not update if the batch size is zero

        # Convert x to torch.float64 for high precision
        x = x.to(dtype=torch.float64).abs()

        if not self.initialized:
            self.abs_mean = x.mean(dim=0)
            self.initialized = True
            return
        
        weights = self.alpha * ((1 - self.alpha) ** torch.arange(x.size(0) - 1, -1, -1, device=self.device))[:, None]
        self.abs_mean = torch.sum(weights * x, dim=0) + (1 - torch.sum(weights, dim=0)) * self.abs_mean

    def get_abs_mean(self):
        return self.abs_mean

    def normalize(self, x):
        if len(x) < 1:
            return x
        abs_mean = self.get_abs_mean()
        normalized_x = x / (abs_mean + 1e-8)
        return normalized_x

    def save(self, path):
        torch.save({'abs_mean': self.abs_mean, \
            'alpha': self.alpha, 'window_size': self.window_size, 'initialized': self.initialized}, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.abs_mean = checkpoint['abs_mean']
        self.alpha = checkpoint['alpha']
        self.window_size = checkpoint['window_size']
        self.initialized = checkpoint['initialized']
