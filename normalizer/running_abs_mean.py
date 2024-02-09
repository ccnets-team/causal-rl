import torch

class RunningAbsMean:
    def __init__(self, num_features, scale, device):
        self.device = device
        self.abs_mean = torch.zeros(num_features, device=self.device, dtype=torch.float64)
        self.count = torch.zeros(num_features, device=self.device, dtype=torch.float64)
        self.num_features = num_features
        self.scale = scale

    def update(self, x, padding_mask=None):
        batch_size, seq_len, feature_size = x.size()
        if batch_size == 0:
            return self  # Do not update if the batch size is zero
        
        x = x.to(dtype=torch.float64)
        if padding_mask is None:
            new_adding = batch_size * seq_len * torch.ones_like(self.count)
            weighted_x = x.abs()  # Use x directly if there is no padding mask
        else:
            mask = padding_mask.to(dtype=torch.float64)
            new_adding = mask.sum(dim = (0, 1))
            weighted_x = mask * x.abs()  # Apply mask to x if padding mask is provided
                
        delta = torch.where(new_adding > 0, torch.sum(weighted_x, dim=(0, 1))/new_adding, torch.zeros_like(new_adding))- self.abs_mean
        new_count = self.count + new_adding
        new_mean = self.abs_mean + delta * torch.where(new_adding > 0, new_adding / new_count, torch.zeros_like(new_adding))

        self.abs_mean = new_mean
        self.count = new_count

        return self

    def get_abs_mean(self):
        return self.abs_mean

    def normalize(self, x):
        if len(x) < 1 or torch.sum(self.count < 1) > 0:
            return x
        abs_mean = self.get_abs_mean().view(1, 1, self.num_features)
        normalized_x = x / (abs_mean / self.scale + 1e-8)
        return normalized_x.to(dtype=x.dtype)

    def save(self, path):
        torch.save({'abs_mean': self.abs_mean, 'count': self.count}, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.abs_mean = checkpoint['abs_mean']
        self.count = checkpoint['count']