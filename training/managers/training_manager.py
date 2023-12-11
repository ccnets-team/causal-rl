import torch.optim as optim
import torch.nn  

class TrainingManager:
    def __init__(self, optimization_params, networks, target_networks):
        self._optimizers = []
        self._schedulers = []
        for network in networks:
            if network is None:
                continue
            opt = optim.Adam(network.parameters(), lr=optimization_params.lr, betas=(0.9, 0.999))
            self._optimizers.append(opt)
            self._schedulers.append(optim.lr_scheduler.StepLR(opt, step_size=optimization_params.step_size, gamma=optimization_params.lr_gamma))
        self._target_networks = target_networks 
        self._networks = networks 
        self._tau = optimization_params.tau
        self.max_grad_norm = optimization_params.max_grad_norm
        
    def get_optimizers(self):
        return self._optimizers

    def get_schedulers(self):
        return self._schedulers

    def update_optimizers(self):
        max_grad_norm = self.max_grad_norm 
        for opt in self._optimizers:
            torch.nn.utils.clip_grad_norm_(opt.param_groups[0]['params'], max_grad_norm)
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