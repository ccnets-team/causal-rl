import torch

class RunningMeanStd:
    def __init__(self, num_features, device):
        self.device = device
        self.mean = torch.zeros(num_features, device=self.device, dtype=torch.float64)
        self.var = torch.ones(num_features, device=self.device, dtype=torch.float64)
        self.count = 0
        self.num_features = num_features

    def update(self, x, padding_mask=None):
        batch_size, seq_len, feature_size = x.size()
        if batch_size == 0:
            return self  # Do not update if the batch size is zero

        x = x.to(dtype=torch.float64)

        new_adding = batch_size * seq_len
        x_mean = x.mean(dim=(0, 1)) 
        x_var = x.var(dim=(0, 1), unbiased=False)
                    
        delta = x_mean - self.mean
        new_count = self.count + new_adding
        new_mean = self.mean + delta * new_adding / new_count

        m_a = self.var * self.count
        m_b = x_var * new_adding
        M2 = m_a + m_b + delta ** 2 * self.count * new_adding / new_count

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
        mean = self.get_mean().view(1, 1, self.num_features)
        var = self.get_var().view(1, 1, self.num_features)
        normalized_x = (x - mean) / (torch.sqrt(var) + 1e-8)
        return normalized_x.to(dtype=x.dtype)

    def save(self, path):
        torch.save({'mean': self.mean, 'var': self.var, 'count': self.count}, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.mean = checkpoint['mean']
        self.var = checkpoint['var']
        self.count = checkpoint['count']