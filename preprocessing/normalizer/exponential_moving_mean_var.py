import torch

class ExponentialMovingMeanVar:
    def __init__(self, num_features, device, alpha=None, window_size=None):
        assert (alpha is None) != (window_size is None), "Specify either alpha or window_size, not both"
        self.device = device

        if window_size is not None:
            alpha = 2 / (window_size + 1)
        self.alpha = alpha
        self.window_size = window_size if window_size is not None else (2 / alpha - 1)

        self.mean = torch.zeros(num_features, device=self.device)
        self.squared_mean = torch.zeros(num_features, device=self.device)
        self.var = torch.ones(num_features, device=self.device)
        self.initialized = False

    def update(self, x):
        if not self.initialized:
            self.mean = torch.mean(x, dim=0)
            self.squared_mean = torch.mean(x ** 2, dim=0)
            self.var = torch.var(x, dim=0, unbiased=False)
            self.initialized = True
            return
        weights = self.alpha * ((1 - self.alpha) ** torch.arange(x.shape[0] - 1, -1, -1, device=self.device))[:, None]
        self.mean = torch.sum(weights * x, dim=0) + (1 - torch.sum(weights)) * self.mean
        self.squared_mean = torch.sum(weights * (x ** 2), dim=0) + (1 - torch.sum(weights)) * self.squared_mean
        self.var = self.squared_mean - self.mean ** 2

    def normalize(self, values):
        if len(values) < 1:
            return values
        normalized_values = (values - self.mean) / (torch.sqrt(self.var) + 1e-8)
        return normalized_values

    def save(self, path):
        torch.save({'mean': self.mean, 'var': self.var, 'initialized': self.initialized}, path)

    def load(self, path):
        data = torch.load(path, map_location=self.device)
        self.mean = data['mean']
        self.var = data['var']
        self.squared_mean = self.var + self.mean ** 2
        self.initialized = data['initialized']