import torch
from preprocessing.normalizer.running_mean_std import RunningMeanStd
from preprocessing.normalizer.exponential_moving_mean_var import ExponentialMovingMeanVar

class HybridMovingMeanVar:
    def __init__(self, batch_size, max_seq_length, num_features, device, alpha=None, window_size=None):
        self.rms = RunningMeanStd(num_features, device)
        self.emmv = ExponentialMovingMeanVar(num_features, device, alpha=alpha, window_size=window_size)
        self.use_emmv = False
        self.rms_frac = 1 
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length

    def _update_fraction(self):
        rms_ratio = (self.batch_size*self.max_seq_length)/max(1, self.rms.count)
        emmv_ratio = self.emmv.alpha
        self.rms_frac = rms_ratio / (rms_ratio + emmv_ratio)
        if self.rms_frac < 1e-8:
            self.use_emmv = True
        return self.rms_frac

    def get_mean(self):
        if self.use_emmv:
            return self.emmv.mean
        else:
            return self.rms.mean * self.rms_frac + self.emmv.mean * (1 - self.rms_frac)

    def get_var(self):
        if self.use_emmv:
            return self.emmv.var
        else:
            return self.rms.var * self.rms_frac + self.emmv.var * (1 - self.rms_frac)
        
    def update(self, x, padding_mask = None):
        self.emmv.update(x, padding_mask)
        if self.use_emmv:
            return 
        self.rms.update(x, padding_mask)
        if not self.use_emmv:
            self._update_fraction()

    def normalize(self, x):
        if len(x) < 1 or self.rms.count < 1 or not self.emmv.initialized:
            return x

        mean = self.get_mean().view(1, 1, -1)
        var = self.get_var().view(1, 1, -1)
        normalized_x = (x - mean) / (torch.sqrt(var) + 1e-8)
        return normalized_x.to(dtype=x.dtype)

    def save(self, path):
        self.rms.save(path)
        self.emmv.save(path)

    def load(self, path):
        self.rms.load(path)
        self.emmv.load(path)