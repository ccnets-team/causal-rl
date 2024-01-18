from torch.utils.hooks import RemovableHandle

class HooksDictWrapper(object):
    def __init__(self, hooks_dict):
        self.hooks_dict = hooks_dict

class HybridHook:
    def __init__(self, state_ratio, action_ratio):
        self.state_ratio = state_ratio
        self.action_ratio = action_ratio
        self.state_hook = None
        self.action_hook = None

    def state_loss_hook(self, grad):
        return grad * self.state_ratio

    def action_loss_hook(self, grad):
        return grad * self.action_ratio

    def hybrid(self, state_tensor, action_tensor):
        # Register hooks to scale gradients
        self.state_tensor = state_tensor
        self.action_tensor = action_tensor
        
        self.state_hook = RemovableHandle(HooksDictWrapper({self.state_tensor: self.remove_state_hook}))
        self.action_hook = RemovableHandle(HooksDictWrapper({self.action_tensor: self.remove_action_hook}))

        self.state_tensor.register_hook(self.state_loss_hook)
        self.action_tensor.register_hook(self.action_loss_hook)

        # Compute combined loss
        combined_loss = self.state_tensor + self.action_tensor
        return combined_loss

    def remove_state_hook(self):
        if self.state_hook is not None:
            self.state_hook.remove()
            self.state_hook = None

    def remove_action_hook(self):
        if self.action_hook is not None:
            self.action_hook.remove()
            self.action_hook = None