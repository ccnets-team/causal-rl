from training.trainer.crl import CausalRL
from training.trainer.ddpg import DDPG
from training.trainer.sac import SAC
from training.trainer.a2c import A2C
from training.trainer.dqn import DQN
from training.trainer.td3 import TD3
from utils.printer import print_rl_params

class RLTrainer:
    def __init__(self, rl_params, trainer_name):
        self.trainer_name = trainer_name
        self.rl_params = rl_params
        self.saved_trainer = None
        print_rl_params(trainer_name, rl_params)

    def load(self):
        return self.saved_trainer

    def initialize(self, env_config, device):
        self.saved_trainer = None
        trainer_map = {
            "crl": CausalRL,
            "ddpg": DDPG,
            "a2c": A2C,
            "dqn": DQN,
            "td3": TD3,
            "sac": SAC,
        }
        self.saved_trainer = trainer_map[self.trainer_name](env_config, self.rl_params, device)
        return self.saved_trainer