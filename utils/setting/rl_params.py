from utils.setting.rl_config import TrainingParameters, AlgorithmParameters, NetworkParameters, OptimizationParameters, ExplorationParameters, MemoryParameters, NormalizationParameters

class RLParameters:
    def __init__(self,
                 training: TrainingParameters = None,
                 algorithm: AlgorithmParameters = None,
                 network: NetworkParameters = None,
                 optimization: OptimizationParameters = None,
                 exploration: ExplorationParameters = None,
                 memory: MemoryParameters = None,
                 normalization: NormalizationParameters = None):
        
        # Initialize RL parameters
        self.training = TrainingParameters() if training is None else training
        self.algorithm = AlgorithmParameters() if algorithm is None else algorithm
        self.network = NetworkParameters() if network is None else network
        self.optimization = OptimizationParameters() if optimization is None else optimization
        self.exploration = ExplorationParameters() if exploration is None else exploration
        self.memory = MemoryParameters() if memory is None else memory
        self.normalization = NormalizationParameters() if normalization is None else normalization

    def __getattr__(self, name):
        # Check if the attribute is part of any of the parameter classes
        for param in [self.training, self.algorithm, self.network, self.optimization, self.exploration, self.memory, self.normalization]:
            if hasattr(param, name):
                return getattr(param, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        # Set attribute if it's one of RLParameters' direct attributes
        if name in ["training", "algorithm", "network", "optimization", "exploration", "memory", "normalization"]:
            super().__setattr__(name, value)
        else:
            # Set attribute in one of the parameter classes
            for param in [self.training, self.algorithm, self.network, self.optimization, self.exploration, self.memory, self.normalization]:
                if hasattr(param, name):
                    setattr(param, name, value)
                    return
            # If the attribute is not found in any of the parameter classes, set it as a new attribute of RLParameters
            super().__setattr__(name, value)

    def __iter__(self):
        yield from [self.training, self.algorithm, self.network, self.optimization, self.exploration, self.memory, self.normalization]