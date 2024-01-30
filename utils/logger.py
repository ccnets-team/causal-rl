from datetime import datetime
import os

METRICS_CATEGORY_MAP = {
    'losses': 'Losses',
    'values': 'Values',
    'errors': 'CoopErrors',
    'costs': 'TransitionCosts'
}
def _log_data(logger, scalar_logs, metrics, step):
    # Log scalar data
    for name, value in scalar_logs.items():
        if value is not None:
            logger.add_scalar(name, value, step)

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
                    logger.add_scalar(log_name, metric_value, step)

def log_train_data(trainer, logger, train_reward_per_step, eval_reward_per_step, train_accumulative_rewards, eval_accumulative_rewards, metrics, step, time_cost):
    learning_rate = trainer.get_lr()

    # Creating a dictionary to log scalar data efficiently
    scalar_logs = {
        "Episode/TrainRewards": train_accumulative_rewards,
        "Episode/EvalRewards": eval_accumulative_rewards,
        "Step/TrainReward": train_reward_per_step,
        "Step/EvalReward": eval_reward_per_step,
        "Step/Time": time_cost,
        "Step/LearningRate": learning_rate,
    }

    _log_data(logger, scalar_logs, metrics, step)

def log_test_data(trainer, logger, test_reward_per_step, test_accumulative_rewards, metrics, step, time_cost):
    learning_rate = trainer.get_lr()

    # Creating a dictionary to log scalar data efficiently
    scalar_logs = {
        "Episode/TestRewards": test_accumulative_rewards,
        "Step/TestReward": test_reward_per_step,
        "Step/Time": time_cost,
        "Step/LearningRate": learning_rate,
    }

    _log_data(logger, scalar_logs, metrics, step)

def get_log_name(log_path):
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f"{log_path}/{current_time}"
    
    # Check if the directory exists, if it does, append a suffix to make it unique
    suffix = 0
    while os.path.isdir(log_dir):
        suffix += 1
        log_dir = f"{log_path}/{current_time}_{suffix}"
    
    return log_dir
