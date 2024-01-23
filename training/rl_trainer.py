from training.trainer.causal_rl import CausalRL
from training.trainer.ddpg import DDPG
from training.trainer.sac import SAC
from training.trainer.a2c import A2C
from training.trainer.dqn import DQN
from training.trainer.td3 import TD3
from utils.printer import print_rl_params

class RLTrainer:
    def __init__(self, rl_params):
        self.trainer_name = rl_params.trainer_name
        self.rl_params = rl_params
        self.saved_trainer = None
        print_rl_params(rl_params.trainer_name, rl_params)

    def load(self):
        return self.saved_trainer

    def initialize(self, env_config, device):
        self.saved_trainer = None
        trainer_map = {
            "causal_rl": CausalRL,
            "ddpg": DDPG,
            "a2c": A2C,
            "dqn": DQN,
            "td3": TD3,
            "sac": SAC,
        }

        # Mapping of trainer names to use_gae_advantage flag
        on_policy_map = {
            'causal_rl': False,
            'ddpg': False,
            'a2c': True,
            'ppo': True,
            'sac': False,
            'td3': False,
            'dqn': False
        }
        
        use_on_policy = on_policy_map.get(self.trainer_name, False)        
        self.rl_params.use_on_policy = use_on_policy
        self.rl_params.use_gae_advantage = use_on_policy
        self.saved_trainer = trainer_map[self.trainer_name](env_config, self.rl_params, device)
        return self.saved_trainer

    @staticmethod
    def create(env_config, rl_params, device):
        # Create and return a new RLTrainer object
        new_rl_trainer = RLTrainer(rl_params)
        return new_rl_trainer.initialize(env_config, device)
        