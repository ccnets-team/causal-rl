
from training.managers.utils.exploration_utility import ExplorationUtils 
from training.managers.utils.normalization_utility import NormalizationUtils

class StrategyManager(NormalizationUtils, ExplorationUtils):
    def __init__(self, env_config, exploration_params, normalization_params, device):  
        NormalizationUtils.__init__(self, env_config, normalization_params, device)
        ExplorationUtils.__init__(self, exploration_params)
    
    def update_strategy(self, samples):
        
        self.update_normalizer(samples)
        self.update_exploration_rate()
        