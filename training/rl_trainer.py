from training.trainer.causal_rl import CausalRL

class RLTrainer:
    def __init__(self, rl_params):
        self.rl_params = rl_params
        self.saved_trainer = None

    def load(self):
        return self.saved_trainer

    def initialize(self, env_config, device):
        self.saved_trainer = CausalRL(env_config, self.rl_params, device)
        return self.saved_trainer

    @staticmethod
    def create(env_config, rl_params, device):
        # Create and return a new RLTrainer object
        new_rl_trainer = RLTrainer(rl_params)
        return new_rl_trainer.initialize(env_config, device)
        