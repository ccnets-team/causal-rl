from .running_mean_std import RunningMeanStd
from .exponential_moving_mean_var import ExponentialMovingMeanVar
import torch

class HybridMovingMeanVar:
    def __init__(self, num_features, device, alpha=None, window_size=None):
        self.rms = RunningMeanStd(num_features, device)
        self.emmv = ExponentialMovingMeanVar(num_features, device, alpha, window_size)
        self.device = device

    @property
    def mean(self):
        emmv_frac = self.rms.count / self.emmv.window_size
        if emmv_frac >= 1:
            return self.emmv.mean
        else:
            return self.rms.mean * (1 - emmv_frac) + self.emmv.mean * emmv_frac

    @property
    def var(self):
        emmv_frac = self.rms.count / self.emmv.window_size
        if emmv_frac >= 1:
            return self.emmv.var
        else:
            return self.rms.var * (1 - emmv_frac) + self.emmv.var * emmv_frac

    def update(self, x):
        self.rms.update(x)
        self.emmv.update(x)

    def normalize(self, values):
        if len(values) < 1:
            return values
        normalized_values = (values - self.mean) / (torch.sqrt(self.var) + 1e-8)
        return normalized_values

    def save(self, path):
        self.rms.save(path + "-rms.pt")
        self.emmv.save(path + "-emmv.pt")

    def load(self, path):
        self.rms.load(path + "-rms.pt")
        self.emmv.load(path + "-emmv.pt")