import torch
# EMA
class RunningAbsMeanScaler:
    def __init__(self, num_features, device, alpha=0.99):
        """
        :param alpha: The decay factor that determines the rate at which the old mean is retained. 
                      High value means more conservative.
        """
        self.device = device
        self.mean = torch.ones(num_features, device=self.device)  # Starting mean as 1
        self.alpha = alpha
        self.max_value = 1.0

    def update(self, x):
        batch_size = x.size(0)
        if batch_size == 0:
            return self  # Do not update if the batch size is zero

        new_mean = x.mean(dim=0)
        self.mean = self.alpha * self.mean + (1 - self.alpha) * new_mean
        if self.mean > self.max_value:
            self.max_value = self.mean
        return self

    def get_max_value(self):
        return self.max_value

    def normalize(self, values):
        if len(values) < 1:
            return values
        max_val = self.get_max_value()
        normalized_values = values / max_val
        return normalized_values

    def save(self, path):
        torch.save({'max_value': self.get_max_value()}, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location="cuda:0")
        self.max_value = checkpoint['max_value']
