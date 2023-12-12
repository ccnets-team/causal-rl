import wandb

def convert_to_dict(rl_params):
    params_dict = rl_params.__dict__.copy()

    for key, value in params_dict.items():
        if hasattr(value, '__dict__'):
            params_dict[key] = value.__dict__

    return params_dict

METRICS_CATEGORY_MAP = {
    'losses': 'Losses',
    'values': 'Values',
    'errors': 'CoopErrors',
    'costs': 'TransitionCosts'
}

def wandb_init(env_config, rl_params):
    wandb.login()
    rl_params_dict = convert_to_dict(rl_params)
    wandb.init(
        project=env_config.env_name,
        save_code = True,
        monitor_gym = False, 
        config= rl_params_dict    
    )

def wandb_end():
    wandb.finish()

def wandb_log_data(trainer, train_reward_per_step, test_reward_per_step, train_accumulative_rewards, test_accumulative_rewards, metrics, step, time_cost):
    epsilon = trainer.get_exploration_rate()
    learning_rate = trainer.get_lr()

    # Creating a dictionary to log scalar data efficiently
    log_data = {
        "Episode/TestRewards": test_accumulative_rewards, 
        'Episode/TrainRewards': train_accumulative_rewards, 
        'Step/TestReward': test_reward_per_step, 
        'Step/TrainReward': train_reward_per_step,
        "Step/Time": time_cost, 
        "Step/LearningRate": learning_rate,
        "Step/ExplorationRate": epsilon
    }

    if metrics is not None:
        # Loop over each metrics category and log each metric with a specific prefix
        for category_name, category_metrics in metrics.__dict__.items():
            # Map the category_name to the new desired name
            mapped_category_name = METRICS_CATEGORY_MAP.get(category_name, category_name.title())
            
            for metric_name, metric_value in category_metrics.items():
                if metric_value is not None:
                    components = metric_name.split('_')

                    if len(components) > 1:  # if there is more than one word in metric_name
                        new_metric_name = components[-2].title()  # select only the previous of the last word and capitalize it
                    else:  # if there is only one word in metric_name
                        new_metric_name = components[0].title()                         
                        
                    log_name = f"{mapped_category_name}/{new_metric_name}"
                    log_data[log_name] = metric_value  # Add the metric to the logging dictionary

    wandb.log(log_data, step=step)  # Log all data including the metrics                    
