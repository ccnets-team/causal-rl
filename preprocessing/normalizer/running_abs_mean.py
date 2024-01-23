import torch

class RunningAbsMean:
    def __init__(self, num_features, device):
        self.device = device
        self.abs_mean = torch.zeros(num_features, device=self.device)
        self.count = 0

    def update(self, x):
        batch_size = x.size(0)
        if batch_size == 0:
            return self  # Do not update if the batch size is zero

        # Convert x to torch.float64 for high precision
        x = x.to(dtype=torch.float64)
        x = x.mean(dim = 0) # 3D -> 2D

        abs_x = torch.abs(x)
        new_count = self.count + batch_size
        new_abs_mean = (self.abs_mean * self.count + abs_x.mean(dim=0) * batch_size) / new_count

        self.abs_mean = new_abs_mean
        self.count = new_count

        return self
    
    def get_abs_mean(self):
        return self.abs_mean

    def normalize(self, x):
        if len(x) < 1 or self.count < 1:
            return x
        abs_mean = self.get_abs_mean().unsqueeze(0)
        normalized_x = x / (abs_mean + 1e-8)
        return normalized_x

    def save(self, path):
        torch.save({'abs_mean': self.abs_mean, 'count': self.count}, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.abs_mean = checkpoint['abs_mean']
        self.count = checkpoint['count']
