import torch

class RunningMinMaxScaler:
    def __init__(self, num_features, device, max_count=1e8):
        self.device = device
        self.max_count = max_count
        self.min_vals = torch.full((num_features,), fill_value=float('inf'), device=self.device)
        self.max_vals = torch.full((num_features,), fill_value=float('-inf'), device=self.device)

    def update(self, x):
        batch_size = x.size(0)
        if batch_size == 0:
            return self  # Do not update if the batch size is zero

        # Update min and max values
        self.min_vals = torch.min(self.min_vals, x.min(dim=0).values)
        self.max_vals = torch.max(self.max_vals, x.max(dim=0).values)

        return self

    def get_min(self):
        return self.min_vals

    def get_max(self):
        return self.max_vals

    def get_range(self):
        return self.max_vals - self.min_vals

    def rescale_counts(self):
        if self.count > self.max_count:
            self.count = self.max_count

    def normalize(self, vals):
        if len(vals) < 1 or self.count < 1:
            return vals
        min_vals = self.get_min()
        range_vals = self.get_range()
        normalized_vals = (vals - min_vals) / (range_vals + 1e-8)
        return normalized_vals

    def save(self, path):
        torch.save({'min': self.get_min(), 'max': self.get_max()}, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location="cuda:0")
        self.min_vals = checkpoint['min']
        self.max_vals = checkpoint['max']