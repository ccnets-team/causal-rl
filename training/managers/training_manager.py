import torch
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler, CyclicLR

LR_CYCLE_SIZE = 20000

class LinearDecayLR(_LRScheduler):
    def __init__(self, optimizer, total_steps, last_epoch=-1):
        self.total_steps = total_steps
        super(LinearDecayLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * (1 - self.last_epoch / self.total_steps) for base_lr in self.base_lrs]

def _apply_gradient_clipping(network, clip_range):
    if clip_range is not None:  
        for param in network.parameters():
            if param.grad is not None:
                param.grad.data = param.grad.data.clamp(-clip_range, clip_range)

def _apply_l2_gradient_normalization(network, max_norm):
    """
    Scales gradients for all parameters in the network based on L2 norm.
    This ensures the norm of the gradients does not exceed the specified max_norm.

    Args:
        network (torch.nn.Module): The neural network whose gradients will be normalized.
        max_norm (float): The maximum allowable L2 norm of the gradients for the parameters.

    Returns:
        Total norm of the parameters (before clipping).
    """
    # Calculate the L2 norm of all gradients.
    if max_norm is not None:
        total_norm = torch.sqrt(sum(torch.sum(param.grad.data ** 2) for param in network.parameters() if param.grad is not None))
        
        # Scale factor to keep the gradient norm at max_norm.
        scale_factor = max_norm / (total_norm + 1e-8)  # Adding epsilon to prevent division by zero.
        
        # If the total norm is larger than the max_norm, scale down gradients
        if scale_factor < 1:
            for param in network.parameters():
                if param.grad is not None:
                    param.grad.data.mul_(scale_factor)

class TrainingManager:
    def __init__(self, networks, target_networks, lr, min_lr, clip_grad_range, max_grad_norm, tau, total_iterations, scheduler_type):
        self._optimizers = []
        self._schedulers = []
        for network in networks:
            if network is None:
                continue
            opt = optim.Adam(network.parameters(), lr=lr, betas=(0.9, 0.999))
            self._optimizers.append(opt)
            if scheduler_type == 'linear':
                self._schedulers.append(LinearDecayLR(opt, total_steps=total_iterations))
            elif scheduler_type == 'exponential':
                lr_decay_ratio = min_lr/lr
                gamma = pow(lr_decay_ratio, 1.0 / total_iterations)
                self._schedulers.append(optim.lr_scheduler.StepLR(opt, step_size=1, gamma=gamma))
            elif scheduler_type == 'cyclic':
                self._schedulers.append(CyclicLR(opt, base_lr=min_lr, max_lr=lr, step_size_up=LR_CYCLE_SIZE // 2, mode='triangular', cycle_momentum=False))
            else:
                raise ValueError(f"Unknown scheduler type: {scheduler_type}")
        self._target_networks = target_networks 
        self._networks = networks 
        self._tau = tau
        self._clip_grad_range = clip_grad_range
        self._max_grad_norm = max_grad_norm

    def get_optimizers(self):
        return self._optimizers

    def get_schedulers(self):
        return self._schedulers

    def clip_gradients(self):
        for net in self._networks:  
            _apply_l2_gradient_normalization(net, self._max_grad_norm)
            _apply_gradient_clipping(net, self._clip_grad_range)

    def update_optimizers(self):
        for opt in self._optimizers:
            opt.step()

    def update_schedulers(self):
        for sc in self._schedulers:
            sc.step()

    def get_lr(self):
        return (self._optimizers[0]).param_groups[0]['lr']

    def update_target_networks(self):
        for target_network, local_network in zip(self._target_networks, self._networks):
            if target_network is None:
                continue
            for target_param, local_param in zip(target_network.parameters(), local_network.parameters()):
                target_param.data.copy_(self._tau * local_param.data + (1.0 - self._tau) * target_param.data)

    def get_networks(self):
        return self._networks
        
    def get_target_networks(self):
        return self._target_networks

    def set_train(self, training):
        for network in self._networks:
            if network is not None: 
                self._set_train(network, training)
        # If target network exists and needs to be set to train/eval mode.
        for target_network in self._target_networks:
            if target_network is not None:
                self._set_train(target_network, training)

    def _set_train(self, network, training):
        network.zero_grad()
        if training:
            network.train()
        else:
            network.eval()