try:
    import wandb
except ImportError:
    wandb = None
from datetime import datetime

now = datetime.now()
formatted_date = now.strftime("%y-%m-%d %H:%M:%S")

def convert_to_dict(rl_params):
    params_dict = rl_params.__dict__.copy()

    for key, value in params_dict.items():
        if hasattr(value, '__dict__'):
            params_dict[key] = value.__dict__

    return params_dict

def sort_key(item):
    key, value = item
    if isinstance(value, str):
        return (0, key)  
    elif isinstance(value, bool):
        return (1, key)  
    else:
        return (2, key)  

METRICS_CATEGORY_MAP = {
    'losses': 'Losses',
    'values': 'Values',
    'errors': 'CoopErrors',
    'costs': 'TransitionCosts'
}

def wandb_init(env_config, rl_params):
    if wandb is None:
        raise RuntimeError("wandb is not installed. Please install wandb to use wandb_init.")
    wandb.login()
    
    env_config_dict = convert_to_dict(env_config)
    rl_params_net_actor, rl_params_net_critic, rl_params_net_rev = convert_to_dict(rl_params.network.actor_params),convert_to_dict(rl_params.network.critic_params),convert_to_dict(rl_params.network.rev_env_params)
    rl_params_dict = convert_to_dict(rl_params)
    rl_params_dict['network']['actor_params'] = rl_params_net_actor
    rl_params_dict['network']['critic_params'] = rl_params_net_critic
    rl_params_dict['network']['rev_env_params'] = rl_params_net_rev
    
    env_config_dict = {k: v for k, v in env_config_dict.items() if isinstance(v, (int, float, str, bool))}
    env_config_dict = dict(sorted(env_config_dict.items(), key=sort_key))
    env_config_dict = {'env_config':env_config_dict}
    
    merged_config_dict = {**env_config_dict, **rl_params_dict}
    
    trainer_name = 'causal_rl'
    
    wandb.init(
        project='causal-rl',
        name= f'{trainer_name}-{env_config.env_name} : {formatted_date}',
        save_code = True,
        monitor_gym = False, 
        config= merged_config_dict
    )
    
    artifact = wandb.Artifact(f'{trainer_name}-{env_config.env_name}', type='model')
    artifact.add_dir(f'./saved/{env_config.env_name}/{trainer_name}', name="saved/")
    wandb.log_artifact(artifact)
    
def wandb_end():
    if wandb is None:
        raise RuntimeError("wandb is not installed. Please install wandb to use wandb_end.")
    wandb.finish()

def _wandb_log_data(log_data, metrics, step):
    if wandb is None:
        print("wandb is not installed. Skipping wandb_log_data.")
        return

    if metrics is not None:
        # Loop over each metrics category and log each metric with a specific prefix
        for category_name, category_metrics in metrics.__dict__.items():
            # Map the category_name to the new desired name
            mapped_category_name = METRICS_CATEGORY_MAP.get(category_name, category_name.title())
            
            for metric_name, metric_value in category_metrics.items():
                if metric_value is not None:
                    components = metric_name.split('_')
                    new_metric_name = components[-2].title() if len(components) > 1 else components[0].title()                         
                    log_name = f"{mapped_category_name}/{new_metric_name}"
                    log_data[log_name] = metric_value  # Add the metric to the logging dictionary

    wandb.log(log_data, step=step)  # Log all data including the metrics

def wandb_log_train_data(trainer, train_reward_per_step, eval_reward_per_step, train_accumulative_rewards, eval_accumulative_rewards, metrics, step, time_cost):
    learning_rate = trainer.get_lr()
    gamma_lambda_learner = trainer.get_gamma_lambda_learner()
    gamma = gamma_lambda_learner.get_gamma()
    input_seq_len = trainer.get_input_seq_len()
    max_seq_len = trainer.get_max_seq_len()
    lambd = gamma_lambda_learner.get_lambdas(seq_range = (0, max_seq_len)).clone().detach().mean()
    
    # Creating a dictionary to log scalar data efficiently
    log_data = {
        'Episode/TrainRewards': train_accumulative_rewards, 
        "Episode/EvalRewards": eval_accumulative_rewards, 
        'Step/TrainReward': train_reward_per_step,
        'Step/EvalReward': eval_reward_per_step, 
        "Step/Time": time_cost, 
        "Step/LearningRate": learning_rate,
        "Step/Gamma": gamma,
        "Step/Lambda": lambd,
        "Step/InputSeqLen": input_seq_len
    }

    _wandb_log_data(log_data, metrics, step)

def wandb_log_test_data(trainer, test_reward_per_step, test_accumulative_rewards, metrics, step, time_cost):
    learning_rate = trainer.get_lr()

    # Creating a dictionary to log scalar data efficiently
    log_data = {
        'Episode/TestRewards': test_accumulative_rewards, 
        'Step/TestReward': test_reward_per_step,
        "Step/Time": time_cost, 
        "Step/LearningRate": learning_rate,
    }

    _wandb_log_data(log_data, metrics, step)
