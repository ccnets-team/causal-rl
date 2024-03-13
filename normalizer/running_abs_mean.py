import torch

class RunningAbsMean:
    def __init__(self, num_features, scale, device, min_abs_mean=1.0, max_abs_mean=None, decay_rate = 1e-3):
        self.device = device
        self.abs_mean = torch.zeros(num_features, device=self.device, dtype=torch.float64)
        self.count = torch.zeros(num_features, device=self.device, dtype=torch.float64)
        self.num_features = num_features
        self.scale = scale
        self.min_abs_mean = min_abs_mean    
        self.max_abs_mean = max_abs_mean    
        self.decay_factor = 1 - decay_rate

    def update(self, x, padding_mask=None, feature_range=None):
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
                
        delta = torch.where(new_adding > 0, torch.sum(weighted_x, dim=(0, 1))/new_adding, self.abs_mean) - self.abs_mean
        new_count = self.count + new_adding
        new_mean = torch.where(new_adding > 0, self.abs_mean + delta * new_adding / new_count, self.abs_mean)

        self.abs_mean = new_mean
        self.count = self.decay_factor * new_count

        return self

    def get_abs_mean(self):
        # Apply the minimum absolute mean threshold if min_abs_mean is not None
        if self.min_abs_mean is not None:
            thresh_min_abs_mean = torch.maximum(self.abs_mean, self.min_abs_mean * torch.ones_like(self.abs_mean))
        else:
            thresh_min_abs_mean = self.abs_mean  # If min_abs_mean is None, do not apply minimum threshold
        
        # Apply the maximum absolute mean threshold if max_abs_mean is not None
        if self.max_abs_mean is not None:
            thresh_abs_mean = torch.minimum(thresh_min_abs_mean, self.max_abs_mean * torch.ones_like(thresh_min_abs_mean))
        else:
            thresh_abs_mean = thresh_min_abs_mean  # If max_abs_mean is None, use the min thresholded value or original abs_mean

        # Now scale the thresholded absolute mean
        scaled_abs_mean = (thresh_abs_mean / self.scale) + 1e-8
        return scaled_abs_mean

    def normalize(self, x, feature_range = None):
        if len(x) < 1 or torch.sum(self.count < 1) > 0:
            return x
        abs_mean = self.get_abs_mean().view(1, 1, self.num_features)
        normalized_x = x / abs_mean
        return normalized_x.to(dtype=x.dtype)

    def save(self, path):
        torch.save({'abs_mean': self.abs_mean, 'count': self.count}, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.abs_mean = checkpoint['abs_mean']
        self.count = checkpoint['count']