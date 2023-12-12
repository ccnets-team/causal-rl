import os

class RLParamsLoader:
    # Initialize RLParamsLoader with environment name and trainer name
    def __init__(self, env_name, trainer_name): 
        self.env_name = env_name
        self.trainer_name = trainer_name

    @staticmethod
    def parse_value(value_str):
        # Parse a value from a string
        value_str = value_str.strip()
        if value_str.isdigit():
            return int(value_str)
        elif "." in value_str:
            try:
                return float(value_str)
            except ValueError:
                pass
        elif value_str == 'True':
            return True
        elif value_str == 'False':
            return False
        else:
            return value_str

    @classmethod
    def load_params(cls, env_name: str, trainer_name: str, train_time=None) -> dict:
        # Load RL parameters from a txt file
        rl_params = {}

        if train_time is None:
            # Get a list of subdirectories inside the trainer_name directory
            trainer_subdirs = [d for d in os.listdir(os.path.join('./log', env_name, trainer_name)) if os.path.isdir(os.path.join('./log', env_name, trainer_name, d))]
            trainer_subdirs.sort()
            latest_subdir = trainer_subdirs[-1]
            directory_path = os.path.join('./log', env_name, trainer_name, latest_subdir)
        else:
            directory_path = os.path.join('./log', env_name, trainer_name, train_time)

        for filename in os.listdir(directory_path):
            if filename.startswith('parameters'):
                file_path = os.path.join(directory_path, filename)
                break
        else:
            raise ValueError(f"No 'parameters' file found in {directory_path}")

        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" in line:
                    key, value_str = line.split(": ")
                    value = cls.parse_value(value_str)
                    rl_params[key] = value

        return rl_params
    
def set_parameters(rl_params, batch_size, num_td_steps, train_frequency, noise_type, network_type, replay_ratio):
    rl_params.training.batch_size = batch_size
    rl_params.algorithm.num_td_steps = num_td_steps
    rl_params.training.train_frequency = train_frequency
    rl_params.exploration.noise_type = noise_type
    rl_params.network.critic_network = network_type
    rl_params.network.actor_network = network_type
    rl_params.network.reverse_env_network = network_type
    rl_params.training.replay_ratio = replay_ratio

def convert_to_dict(rl_params):
    params_dict = rl_params.__dict__.copy()

    for key, value in params_dict.items():
        if hasattr(value, '__dict__'):
            params_dict[key] = value.__dict__

    return params_dict