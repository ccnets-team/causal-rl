import torch

class RunningMeanStd:
    def __init__(self, num_features, device):
        self.device = device
        self.mean = torch.zeros(num_features, device=self.device, dtype=torch.float64)
        self.var = torch.ones(num_features, device=self.device, dtype=torch.float64)
        self.count = 0

    def update(self, x, padding_mask=None):
        batch_size, seq_len, feature_size = x.size()
        if batch_size == 0:
            return self  # Do not update if the batch size is zero

        new_adding = 0
        x = x.to(dtype=torch.float64)

        if padding_mask is None:
            new_adding = batch_size * seq_len
            x = x.view(-1, feature_size)  # Use x directly if there is no padding mask
        else:
            mask = padding_mask.to(dtype=torch.float64)
            # Ensure the mask is expanded to match x's dimensions for masking
            expanded_mask = mask.expand_as(x)
            # Filter x by the mask and then flatten; only include values where mask > 0
            x = x[expanded_mask > 0]
            x = x.view(-1, feature_size)
            new_adding = mask.sum()
                    
        delta = x.mean(dim=0) - self.mean
        new_count = self.count + new_adding
        new_mean = self.mean + delta * new_adding / new_count

        m_a = self.var * self.count
        m_b = x.var(dim=0, unbiased=False) * new_adding
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
        mean = self.get_mean().view(1, 1, -1)
        var = self.get_var().view(1, 1, -1)
        normalized_x = (x - mean) / (torch.sqrt(var) + 1e-8)
        return normalized_x.to(dtype=x.dtype)

    def save(self, path):
        torch.save({'mean': self.mean, 'var': self.var, 'count': self.count}, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.mean = checkpoint['mean']
        self.var = checkpoint['var']
        self.count = checkpoint['count']