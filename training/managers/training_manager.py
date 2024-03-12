import torch
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler, CyclicLR
from ..learnable_td import LearnableTD

LR_CYCLE_SIZE = 20000
STEPS_100K = 100000  # Represents the number of steps over which decay is applied

class LinearDecayLR(_LRScheduler):
    def __init__(self, optimizer, total_steps, decay_rate_100k, last_epoch=-1):
        self.total_steps = total_steps
        # Calculate the final learning rate multiplier considering the total steps
        self.final_lr_multiplier = pow(decay_rate_100k, total_steps / STEPS_100K)
        super(LinearDecayLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        # Calculate the proportional step for the current epoch relative to total_steps
        proportional_step = min(self.last_epoch / self.total_steps, 1)
        # Linearly interpolate the learning rate towards the final_lr_multiplier
        lr_decay = 1 - proportional_step + (self.final_lr_multiplier * proportional_step)
        return [base_lr * lr_decay for base_lr in self.base_lrs]

class TrainingManager:
    def __init__(self, networks, target_networks, learning_param_list, tau, total_iterations):

        self._optimizers = []
        self._schedulers = []
        self._clip_grad_ranges = []
        self._max_grad_norms = []
        for network, learning_param in zip(networks, learning_param_list):
            # Extract learning parameters
            lr = float(learning_param['lr'])
            decay_rate_100k = float(learning_param['decay_rate_100k'])
            scheduler_type = learning_param['scheduler_type']
            clip_grad_range = learning_param['clip_grad_range']
            max_grad_norm = learning_param['max_grad_norm']

            if network is None:
                continue

            # Initialize the optimizer
            opt = optim.Adam(network.parameters(), lr=lr, betas=(0.9, 0.999))
            self._optimizers.append(opt)


            if scheduler_type == 'linear':
                # Set up LinearDecayLR with the corrected final learning rate multiplier
                self._schedulers.append(LinearDecayLR(opt, total_steps=total_iterations, decay_rate_100k=decay_rate_100k))
            # Setup for other scheduler types as before
            elif scheduler_type == 'exponential':
                # Calculate an appropriate gamma for exponential decay
                gamma = pow(decay_rate_100k, 1/STEPS_100K)
                self._schedulers.append(optim.lr_scheduler.StepLR(opt, step_size = 1, gamma=gamma))
            elif scheduler_type == 'cyclic':
                # Adjust CyclicLR setup without using min_lr
                base_lr = lr * decay_rate_100k * STEPS_100K/total_iterations # Assuming some relationship for demonstration
                max_lr = lr
                self._schedulers.append(CyclicLR(opt, base_lr=base_lr, max_lr=max_lr, step_size_up=LR_CYCLE_SIZE // 2, mode='triangular', cycle_momentum=False))
            else:
                raise ValueError(f"Unknown scheduler type: {scheduler_type}")

            self._clip_grad_ranges.append(clip_grad_range)
            self._max_grad_norms.append(max_grad_norm)

        self._target_networks = target_networks 
        self._networks = networks 
        self._tau = tau

    def get_optimizers(self):
        return self._optimizers

    def get_schedulers(self):
        return self._schedulers

    def clip_gradients(self):
        for idx, net in enumerate(self._networks):
            # Check if net is an instance of LearnableTD
            clip_grad_range = self._clip_grad_ranges[idx]
            max_grad_norm = self._max_grad_norms[idx]
            if isinstance(net, LearnableTD):
                # Specific handling for LearnableTD instances
                if max_grad_norm is not None:
                    net.clip_grad_norm_(max_grad_norm)
                if clip_grad_range is not None:
                    if net.raw_gamma.grad is not None:
                        net.raw_gamma.grad.clamp_(-clip_grad_range, clip_grad_range)
                    if net.raw_lambd.grad is not None:
                        net.raw_lambd.grad.clamp_(-clip_grad_range, clip_grad_range)
            else:
                # Handling for other network types
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
                if clip_grad_range is not None:
                    torch.nn.utils.clip_grad_value_(net.parameters(), clip_grad_range)

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