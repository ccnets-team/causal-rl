import torch

class RunningAbsMean:
    def __init__(self, num_features, device, use_min_threshold=True):
        self.device = device
        self.use_min_threshold = use_min_threshold
        self.abs_mean = torch.zeros(num_features, device=self.device, dtype=torch.float64)
        self.count = 0

    def update(self, x, padding_mask=None):
        batch_size = x.size(0)
        if batch_size == 0:
            return self  # Do not update if the batch size is zero
        
        x = x.to(dtype=torch.float64)
        x = x.view(-1, x.shape[-1]) # 3D -> 2D
        
        delta = x.abs().mean(dim=0) - self.abs_mean
        new_count = self.count + batch_size
        new_mean = self.abs_mean + delta * batch_size / new_count

        self.abs_mean = new_mean
        self.count = new_count

        return self

    def get_abs_mean(self):
        if self.use_min_threshold:
            return torch.min(self.abs_mean + 1e-8, torch.ones_like(self.abs_mean))
        else:
            return self.abs_mean + 1e-8

    def normalize(self, x):
        if len(x) < 1 or self.count < 1:
            return x
        abs_mean = self.get_abs_mean().view(1, 1, -1)
        normalized_x = x / abs_mean
        return normalized_x.to(dtype=x.dtype)

    def save(self, path):
        torch.save({'abs_mean': self.abs_mean, 'count': self.count}, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.abs_mean = checkpoint['abs_mean']
        self.count = checkpoint['count']