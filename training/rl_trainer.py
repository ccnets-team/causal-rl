from training.causal_rl import CausalRL

class RLTrainer:
    def __init__(self, rl_params, trainer_name = "causal_rl"):
        self.trainer_name = trainer_name
        self.rl_params = rl_params
        self.saved_trainer = None

    def load(self):
        return self.saved_trainer

    def initialize(self, env_config, device):
        self.saved_trainer = None
        trainer_map = {
            "causal_rl": CausalRL,
            # Other methods are planned but not implemented in this release version:
            # "ddpg": DDPG,
            # "a2c": A2C,
            # "dqn": DQN,
            # "td3": TD3,
            # "sac": SAC,
        }
        
        self.saved_trainer = trainer_map[self.trainer_name](env_config, self.rl_params, device)
        return self.saved_trainer

    @staticmethod
    def create(env_config, rl_params, device):
        # Create and return a new RLTrainer object
        new_rl_trainer = RLTrainer(rl_params)
        return new_rl_trainer.initialize(env_config, device)
        