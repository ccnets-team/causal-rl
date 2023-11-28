# Welford's method to update the mean and variance online, has a normalization method to avoid numeric instability, and even provides save and load methods. 

import torch

class RunningZStandardizer:
    def __init__(self, num_features, device, max_count = 1e9):
        self.device = device
        self.mean = torch.zeros(num_features, device=self.device)
        self.M2 = torch.zeros(num_features, device=self.device)
        self.max_count = max_count
        self.count = 0

    def update(self, x):
        batch_size = x.size(0)
        if batch_size == 0:
            return self  # Do not update if the batch size is zero

        delta = x.mean(dim=0) - self.mean
        # Regular accumulation strategy before reaching max_count
        self.mean += delta * batch_size / (self.count + batch_size)
        self.count += batch_size

        # Note: the computation for M2 is common to both strategies and can be done outside the conditional blocks
        delta2 = x.mean(dim=0) - self.mean
        self.M2 += delta * delta2 * batch_size + ((x - self.mean) ** 2).sum(dim=0)

        # If count exceeds max_count, apply decay
        if self.count > self.max_count:
            self.M2 *= self.max_count/self.count
            self.count = self.max_count
        return self

    def get_mean(self):
        return self.mean

    def get_std(self):
        return torch.sqrt(self.M2 / max(self.count, 1))
                
    def normalize(self, values):
        if len(values) < 1:
            return values
        mean = self.get_mean()
        std = self.get_std()
        normalized_values = (values - mean) / (std + 1e-8)
        return normalized_values

    def save(self, path):
        torch.save({'mean': self.get_mean(), 'M2': self.M2, 'count': self.count}, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location="cuda:0")
        self.mean = checkpoint['mean']
        self.M2 = checkpoint['M2']
        self.count = checkpoint['count']
