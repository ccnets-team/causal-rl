import torch
from torch.utils.hooks import RemovableHandle

class HooksDictWrapper(object):
    def __init__(self, hooks_dict):
        self.hooks_dict = hooks_dict

class HybridHook:
    def __init__(self, state_size_ratio, action_size_ratio):
        self.state_size_ratio = state_size_ratio
        self.action_size_ratio = action_size_ratio
        self.state_ratio = None
        self.action_ratio = None
        self.state_hook_handle = None
        self.action_hook_handle = None

    def __enter__(self):
        # This method is called when entering the context
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # This method is called when exiting the context
        self.remove_hooks()

    def state_loss_hook(self, grad):
        return grad * self.state_ratio

    def action_loss_hook(self, grad):
        return grad * self.action_ratio

    def compute_ratios(self, state_tensor, action_tensor):
        """
        Compute the ratios based on the state and action tensors.

        :param state_tensor: Tensor representing the state.
        :param action_tensor: Tensor representing the action.
        :return: None, but updates state_ratio and action_ratio of the class.
        """
        state_scale = state_tensor.detach().abs()
        action_scale = action_tensor.detach().abs()

        total_scale = state_scale + action_scale
        action_scale_ratio = torch.where(total_scale == 0, 0.5 * torch.ones_like(total_scale), action_scale / total_scale)
        state_scale_ratio = 1 - action_scale_ratio

        # Adjust the ratios by the initial state and action ratios
        # state_ratio = self.state_size_ratio * state_scale_ratio
        # action_ratio = self.action_size_ratio * action_scale_ratio
        state_ratio = state_scale_ratio
        action_ratio = action_scale_ratio

        total_ratio = state_ratio + action_ratio
        self.state_ratio = state_ratio / total_ratio
        self.action_ratio = action_ratio / total_ratio

    def hybrid(self, state_tensor, action_tensor):
        # Call the separate function to compute ratios
        self.compute_ratios(state_tensor, action_tensor)

        # Register hooks to scale gradients
        self.state_tensor = state_tensor
        self.action_tensor = action_tensor
        self.state_tensor.register_hook(self.state_loss_hook)
        self.action_tensor.register_hook(self.action_loss_hook)

        # Compute combined loss
        combined_loss = self.state_tensor + self.action_tensor
        return combined_loss

    def remove_hooks(self):
        if self.state_hook_handle is not None:
            self.state_hook_handle.remove()
            self.state_hook_handle = None
        if self.action_hook_handle is not None:
            self.action_hook_handle.remove()
            self.action_hook_handle = None