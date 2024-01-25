from collections.abc import Iterable
from utils.setting.rl_params import RLParameters
from utils.structure.env_config import EnvConfig
from datetime import datetime

def print_iter(epoch, replay_ratio, iters, len_dataloader, et):
    print('[%d/%d][%d/%d][Time %.2f]'
        % (epoch, replay_ratio, iters, len_dataloader, et))
    
def print_lr(optimizers):
    if isinstance(optimizers, Iterable):
        cur_lr = [opt.param_groups[0]['lr'] for opt in optimizers]
        name_opt = [type(opt).__name__ for opt in optimizers]
        
        if len(set(cur_lr)) == 1 and len(set(name_opt)) == 1:
            print('Opt-{0} lr: {1}'.format(name_opt[0], cur_lr[0]))
        else:
            for i, (lr, opt_name) in enumerate(zip(cur_lr, name_opt)):
                print('Opt-{0} lr_{1}: {2}'.format(opt_name, i, lr))
    else:
        opt = optimizers
        cur_lr = opt.param_groups[0]['lr']
        name_opt = type(opt).__name__ 
        print('Opt-{0} lr: {1}'.format(name_opt, cur_lr))

def _print_metrics(label, **kwargs):
    if all(value is None for value in kwargs.values()):
        return  # Return early if all values are None

    print(label, end=' ')
    for key, value in kwargs.items():
        if value is not None:
            components = key.split('_')
            if len(components) > 1:  # if there is more than one word in metric_name
                print_key = components[-2].title()  # select only the previous of the last word and capitalize it
            else:  # if there is only one word in metric_name
                print_key = components[0].title()      
                
            print(f"{print_key}: {value:.4f}", end='\t')
    print()  # print a newline at the end
    

def print_step(trainer, memory, episode, time_cost):
    max_steps = trainer.exploration_params.max_steps
    if memory is not None:
        buffer_size = len(memory)
    else:
        buffer_size = 0
    
    # Display episode and buffer info
    print(f"[{episode}/{max_steps}] \tbuffer_size: {buffer_size}")
    
    # Display optimizer and learning rate info
    optimizer_type = type(trainer.get_optimizers()[0]).__name__
    learning_rate = trainer.get_lr()
    print(f"Opt-{optimizer_type} lr: {learning_rate}")
    # Display time info
    print(f"Time for steps is {time_cost:.2f} sec")
    
        
def print_scores(train_reward_per_step, test_reward_per_step, train_accumulative_rewards, test_accumulative_rewards):
    if (train_reward_per_step is not None) and (test_reward_per_step is not None):
        print(f"TrainStepReward: {train_reward_per_step:.4f} \tTestStepReward: {test_reward_per_step:.4f}")
    if (train_accumulative_rewards is not None) and (test_accumulative_rewards is not None):
        print(f"TrainEpisodeRewards: {train_accumulative_rewards:.4f} \tTestEpisodeRewards: {test_accumulative_rewards:.4f}")
        
def print_metrics(metrics):
    # Display training and testing details
    if metrics is not None:
        # Usage
        _print_metrics("Values:", **metrics.values.data)
        _print_metrics("Losses:", **metrics.losses.data)
        _print_metrics("CoopErrors:", **metrics.errors.data)
        _print_metrics("TransitionCosts:", **metrics.costs.data)
        
def print_env_specs(env_config: EnvConfig):
    print(f"Environment Specifications for {env_config.env_name}\n")
    print(f"num_environments: {env_config.num_environments}, num_agents: {env_config.num_agents}, samples_per_step: {env_config.samples_per_step}")
    print(f"state_size: {env_config.state_size}, action_size: {env_config.action_size}")
    print(f"use_discrete: {env_config.use_discrete}\n")

def print_params(obj: object, title: str):
    print(f"{title}:")
    for attr, value in obj.__dict__.items():
        if hasattr(value, '__dict__'):
            # Print the attribute name and type of the object instead of the memory address
            print(f"{attr} (type: {type(value).__name__}): ", end='')
            for attr2, value2 in value.__dict__.items():
                # Filter out unnecessary attributes
                if attr2 not in ['__module__', '__init__', 'forward', '__doc__']:
                    print(f"{attr2}: {value2}", end=', ')
        else:
            # Filter out unnecessary attributes for the main object as well
            if attr not in ['__module__', '__init__', 'forward', '__doc__']:
                print(f"{attr}: {value}", end=', ')
    print("\n")  # to print a newline at the end

def save_params_to_file(obj: object, file, is_nested=False):
    items = []
    for attr, value in obj.__dict__.items():
        if hasattr(value, '__dict__'):
            # Filter out unnecessary attributes for nested objects
            if attr not in ['__module__', '__init__', 'forward', '__doc__']:
                nested_items = save_params_to_file(value, file, is_nested=True)
                items.append(f"{attr} (type: {type(value).__name__}): {nested_items}")
        else:
            # Filter out unnecessary attributes for the main object
            if attr not in ['__init__', 'forward', '__doc__']:
                if attr is '__module__':
                    items.append(f"{value}")
                else:
                    items.append(f"{attr}: {value}")
    return ', '.join(items) if is_nested else file.write(', '.join(items) + '\n')

def save_training_parameters_to_file(log_dir_path: str, trainer_name: str, rl_params: RLParameters):
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{log_dir_path}/parameters_{current_time}.txt"
    
    with open(filename, 'w') as file:
        file.write(f"Trainer Name: {trainer_name}\n\n")  # prints the trainer name at the beginning of the file
        # Iterate over the attributes of rl_params and save them to file
        for attr, value in rl_params.__dict__.items():
            file.write(f"{attr} (type: {type(value).__name__}): ")
            save_params_to_file(value, file)
            file.write("\n")  # prints a newline character between different param_objects
