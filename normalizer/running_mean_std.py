import torch

class RunningMeanStd:
    def __init__(self, num_features, scale, device, decay_rate = 1e-5):
        self.device = device
        self.mean = torch.zeros(num_features, device=self.device, dtype=torch.float64)
        self.var = torch.ones(num_features, device=self.device, dtype=torch.float64)
        self.count = torch.zeros(num_features, device=self.device, dtype=torch.float64)
        self.num_features = num_features
        self.scale = scale
        self.decay_factor = 1 - decay_rate

    def update(self, x, padding_mask=None, feature_range = None):
        batch_size, seq_len, feature_size = x.size()
        if batch_size == 0:
            return self  # Do not update if the batch size is zero

        x = x.to(dtype=torch.float64)

        if padding_mask is None:
            new_adding = batch_size * seq_len
            x_mean = x.mean(dim=(0, 1)) 
            x_var = x.var(dim=(0, 1), unbiased=False)
        else:
            mask = padding_mask.to(dtype=torch.float64)
            new_adding = mask.sum(dim=(0, 1)).clamp(min=1)
            x_mean = (mask*x).sum(dim=(0, 1))/new_adding
            # Calculate variance for masked data
            # Subtract the mean from x where padding_mask64 is 1, square it, then sum and average it across the valid (non-masked) entries
            diff_sq = (mask * (x - x_mean.unsqueeze(0).unsqueeze(0))) ** 2
            x_var = diff_sq.sum(dim=(0, 1)) / new_adding

        # Define the update range
        start_idx, end_idx = (0, feature_size) if feature_range is None else feature_range
        delta = x_mean - self.mean[start_idx:end_idx]
        new_count = self.count[start_idx:end_idx] + new_adding

        # Update mean and variance
        new_mean = self.mean[start_idx:end_idx] + delta * new_adding / new_count
        m_a = self.var[start_idx:end_idx] * self.count[start_idx:end_idx]
        m_b = x_var * new_adding
        M2 = m_a + m_b + delta ** 2 * self.count[start_idx:end_idx] * new_adding / new_count

        # Apply updates
        self.mean[start_idx:end_idx] = new_mean
        self.var[start_idx:end_idx] = M2 / new_count
        self.count[start_idx:end_idx] = self.decay_factor * new_count
                    
        return self

    def get_mean(self):
        return self.mean

    def get_var(self):
        return self.var

    def normalize(self, x, feature_range = None):
        if len(x) < 1:
            return x
        mean = self.get_mean().view(1, 1, self.num_features)
        var = self.get_var().view(1, 1, self.num_features)
        
        if feature_range is None:
            normalized_x = (x - mean) / (torch.sqrt(var) / self.scale + 1e-8)
        else:
            (start_idx, end_idx) = feature_range
            normalized_x = (x - mean[:, :, start_idx:end_idx]) / (torch.sqrt(var[:, :, start_idx:end_idx]) / self.scale + 1e-8)
        return normalized_x.to(dtype=x.dtype)

    def save(self, path):
        torch.save({'mean': self.mean, 'var': self.var, 'count': self.count}, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.mean = checkpoint['mean']
        self.var = checkpoint['var']
        self.count = checkpoint['count']