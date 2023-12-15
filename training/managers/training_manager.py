import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler

class LinearDecayLR(_LRScheduler):
    def __init__(self, optimizer, total_steps, last_epoch=-1):
        self.total_steps = total_steps
        super(LinearDecayLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * (1 - self.last_epoch / self.total_steps) for base_lr in self.base_lrs]

def _apply_gradient_clipping(network, clip_range):
    for param in network.parameters():
        if param.grad is not None:
            param.grad.data = param.grad.data.clamp(-clip_range, clip_range)

class TrainingManager:
    def __init__(self, optimization_params, total_iterations, networks, target_networks):
        self._optimizers = []
        self._schedulers = []
        for network in networks:
            if network is None:
                continue
            opt = optim.Adam(network.parameters(), lr=optimization_params.lr, betas=(0.9, 0.999))
            self._optimizers.append(opt)
            self._schedulers.append(LinearDecayLR(opt, total_steps=total_iterations))
        self._target_networks = target_networks 
        self._networks = networks 
        self._tau = optimization_params.tau
        self.clip_grad_range = optimization_params.clip_grad_range
        
    def get_optimizers(self):
        return self._optimizers

    def get_schedulers(self):
        return self._schedulers

    def clip_gradients(self):
        if self.clip_grad_range is not None:  
            for net in self._networks:
                _apply_gradient_clipping(net, self.clip_grad_range)

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