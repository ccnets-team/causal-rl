import torch
import random
import numpy as np

def set_seed(seed = 0):
    """
    fix seed to control any randomness from a code
    (enable stability of the experiments' results.)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    