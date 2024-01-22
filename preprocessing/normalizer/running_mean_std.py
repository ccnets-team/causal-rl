import torch

class RunningMeanStd:
    def __init__(self, num_features, device):
        self.device = device
        self.mean = torch.zeros(num_features, device=self.device, dtype=torch.float64)
        self.var = torch.ones(num_features, device=self.device, dtype=torch.float64)
        self.count = 0

    def update(self, x):
        batch_size = x.size(0)
        if batch_size == 0:
            return self  # Do not update if the batch size is zero
        x = x.to(dtype=torch.float64)

        delta = x.mean(dim=0) - self.mean
        new_count = self.count + batch_size
        new_mean = self.mean + delta * batch_size / new_count

        m_a = self.var * self.count
        m_b = x.var(dim=0, unbiased=False) * batch_size
        M2 = m_a + m_b + delta ** 2 * self.count * batch_size / new_count

        self.mean = new_mean
        self.var = M2 / new_count
        self.count = new_count

        return self

    def get_mean(self):
        return self.mean

    def get_var(self):
        return self.var

    def normalize(self, x):
        if len(x) < 1 or self.count < 1:
            return x
        mean = self.get_mean()
        var = self.get_var()
        normalized_x = (x - mean) / (torch.sqrt(var) + 1e-8)
        return normalized_x.to(dtype=x.dtype)

    def save(self, path):
        torch.save({'mean': self.mean, 'var': self.var, 'count': self.count}, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.mean = checkpoint['mean']
        self.var = checkpoint['var']
        self.count = checkpoint['count']