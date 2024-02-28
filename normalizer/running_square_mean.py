import torch

class RunningSquareMean:
    def __init__(self, num_features, scale, device, min_square_mean=1.0, max_square_mean=None, decay_rate = 1e-3):
        self.device = device
        self.square_mean = torch.zeros(num_features, device=self.device, dtype=torch.float64)
        self.count = torch.zeros(num_features, device=self.device, dtype=torch.float64)
        self.num_features = num_features
        self.scale = scale
        self.min_square_mean = min_square_mean    
        self.max_square_mean = max_square_mean    
        self.decay_factor = 1 - decay_rate

    def update(self, x, padding_mask=None):
        batch_size, seq_len, feature_size = x.size()
        if batch_size == 0:
            return self  # Do not update if the batch size is zero
        
        x = x.to(dtype=torch.float64)
        if padding_mask is None:
            new_adding = batch_size * seq_len * torch.ones_like(self.count)
            weighted_x = x.square()  # Use x directly if there is no padding mask
        else:
            mask = padding_mask.to(dtype=torch.float64)
            new_adding = mask.sum(dim = (0, 1))
            weighted_x = mask * x.square()  # Apply mask to x if padding mask is provided
                
        delta = torch.where(new_adding > 0, torch.sum(weighted_x, dim=(0, 1))/new_adding, self.square_mean) - self.square_mean
        new_count = self.count + new_adding
        new_mean = torch.where(new_adding > 0, self.square_mean + delta * new_adding / new_count, self.square_mean)

        self.square_mean = new_mean
        self.count = self.decay_factor* new_count

        return self

    def get_square_mean(self):
        # Apply the minimum absolute mean threshold if min_abs_mean is not None
        if self.min_square_mean is not None:
            thresh_min_square_mean = torch.maximum(self.square_mean, self.min_square_mean * torch.ones_like(self.square_mean))
        else:
            thresh_min_square_mean = self.square_mean  # If min_square_mean is None, do not apply minimum threshold
        
        # Apply the maximum squareolute mean threshold if max_square_mean is not None
        if self.max_square_mean is not None:
            thresh_square_mean = torch.minimum(thresh_min_square_mean, self.max_square_mean * torch.ones_like(thresh_min_square_mean))
        else:
            thresh_square_mean = thresh_min_square_mean  # If max_square_mean is None, use the min thresholded value or original square_mean

        # Now scale the thresholded squareolute mean
        scaled_square_mean = (torch.sqrt(thresh_square_mean + 1e-8) / self.scale)
        return scaled_square_mean
    
    def normalize(self, x):
        if len(x) < 1 or torch.sum(self.count < 1) > 0:
            return x
        square_mean = self.get_square_mean().view(1, 1, self.num_features)
        normalized_x = x / square_mean
        return normalized_x.to(dtype=x.dtype)

    def save(self, path):
        torch.save({'square_mean': self.square_mean, 'count': self.count}, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.square_mean = checkpoint['square_mean']
        self.count = checkpoint['count']