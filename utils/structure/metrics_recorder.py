import numpy as np
import torch

def convert_to_float(kwargs, keys_list):
    return {
        key: float(value.mean()) if isinstance(value, (torch.Tensor, np.ndarray)) else value
        for key, value in kwargs.items() if key in keys_list
    }

from collections import OrderedDict

def convert_to_float(kwargs, keys_list):
    return OrderedDict(
        (key, float(value.mean()) if isinstance(value, (torch.Tensor, np.ndarray)) else value)
        for key, value in kwargs.items() if key in keys_list
    )

class MetricsBase:
    def __init__(self, data, keys_list):
        self.data = convert_to_float(data, keys_list)

    def items(self):
        return self.data.items()
            
    def __iadd__(self, other):
        all_keys = list(self.data.keys()) + [key for key in other.data.keys() if key not in self.data]  # Maintain order
        for key in all_keys:
            self_val = self.data.get(key)
            other_val = other.data.get(key)
            
            if self_val is None and other_val is not None:
                self.data[key] = other_val  # Maintain order when adding new items
            elif self_val is not None and other_val is not None:
                self.data[key] += other_val
        return self

    def __itruediv__(self, divisor):
        if divisor == 0:
            raise ValueError("Cannot divide by zero")
        for key, value in self.data.items():
            if value is not None:
                self.data[key] /= divisor
        return self
    
    def __iter__(self):
        yield from self.data.values()


class ValueMetrics(MetricsBase):
    def __init__(self, **kwargs):
        super().__init__(kwargs, ['train_reward', 'estimated_value', 'expected_value', 'advantage'])

        
class LossMetrics(MetricsBase):
    def __init__(self, **kwargs):
        super().__init__(kwargs, ['value_loss', 'policy_loss', 'critic_loss', 'critic_loss1', 'critic_loss2', 'actor_loss', 'revEnv_loss'])


class TransitionCostMetrics(MetricsBase):
    def __init__(self, **kwargs):
        super().__init__(kwargs, ['forward_cost', 'reverse_cost', 'recurrent_cost'])


class CoopErrorMetrics(MetricsBase):
    def __init__(self, **kwargs):
        super().__init__(kwargs, ['coop_critic_error', 'coop_actor_error', 'coop_revEnv_error'])

class TrainingMetrics:
    def __init__(self, 
                 values: ValueMetrics=None, 
                 losses: LossMetrics=None,
                 errors: CoopErrorMetrics=None, 
                 costs: TransitionCostMetrics=None 
                 ):

        self.values = values or ValueMetrics()
        self.losses = losses or LossMetrics()
        self.errors = errors or CoopErrorMetrics()
        self.costs = costs or TransitionCostMetrics()

    def __iadd__(self, other):
        self.values += other.values
        self.losses += other.losses
        self.errors += other.errors
        self.costs += other.costs
        return self

    def __itruediv__(self, divisor):
        self.values /= divisor
        self.losses /= divisor
        self.errors /= divisor
        self.costs /= divisor
        return self
    
    def __iter__(self):
        yield from self.values.items()
        yield from self.losses.items()
        yield from self.errors.items()
        yield from self.costs.items()


class MetricsTracker:
    def __init__(self, n_steps):
        self.n_steps = n_steps
        self.steps = [TrainingMetrics() for _ in range(n_steps)]
        self.valid = np.full(n_steps, False, dtype=np.bool8)  # Boolean array to keep track of valid entries
        self.index = 0  # Position in which to add new values

    def add_step(self, step):
        # Assume step is of type TrainMetrics
        if step is None:
            return  
        self.steps[self.index] = step
        self.valid[self.index] = True
        self.index = (self.index + 1) % self.n_steps

    def compute_average(self):
        avg_metrics = TrainingMetrics()
        valid_count = np.sum(self.valid)

        if valid_count < 1:
            return None

        for i, step in enumerate(self.steps):
            if self.valid[i]:  # Check for validity using the boolean array
                avg_metrics += step
        
        avg_metrics /= valid_count
        return avg_metrics


def create_training_metrics(**kwargs):
    """Helper method to create TrainingMetrics object."""
    values = ValueMetrics(
        estimated_value=kwargs.get('estimated_value'),
        expected_value=kwargs.get('expected_value'),
        advantage=kwargs.get('advantage')
    )
    
    losses = LossMetrics(
        value_loss=kwargs.get('value_loss'),
        critic_loss=kwargs.get('critic_loss'),
        actor_loss=kwargs.get('actor_loss'),
        revEnv_loss=kwargs.get('revEnv_loss')
    )
    
    costs = TransitionCostMetrics(
        forward_cost=kwargs.get('forward_cost'),
        reverse_cost=kwargs.get('reverse_cost'),
        recurrent_cost=kwargs.get('recurrent_cost')
    )
    
    errors = CoopErrorMetrics(
        coop_critic_error=kwargs.get('coop_critic_error'),
        coop_actor_error=kwargs.get('coop_actor_error'),
        coop_revEnv_error=kwargs.get('coop_revEnv_error')
    )
    
    return TrainingMetrics(
        values=values,
        losses=losses,
        errors=errors,
        costs=costs
    )    

class RewardTracker:
    def __init__(self, n_steps):
        self.n_steps = n_steps
        self.steps = np.full(n_steps, np.nan, dtype=np.float32)
        self.counts = np.full(n_steps, 0, dtype=np.int32)
        self.index = 0
        
        self.best_reward = -np.inf
        self.best_period = False
        self.accumulative_rewards = {}
        self.episode_accumulative_rewards = {}  # Maintain as dict to store per agent
        self.episode_counts = {}  # Track the number of episodes ended for each agent
                
    def is_best_record_period(self):
        return self.best_period

    def _add_rewards(self, env_ids, agent_ids, rewards, dones):
        if rewards is None:
            return 
        self.steps[self.index] = 0 if len(rewards) == 0 else np.mean(np.array(rewards, np.float32))
        self.counts[self.index] = len(rewards)

        for env_id, agent_id, reward, done in zip(env_ids, agent_ids, rewards, dones):
            key = (env_id, agent_id)
            self.accumulative_rewards[key] = self.accumulative_rewards.get(key, 0) + reward
            if done:
                self.episode_accumulative_rewards[key] = self.accumulative_rewards[key]
                del self.accumulative_rewards[key]  # Reset accumulative reward for the episode that ended
                
                # Increment episode count for the agent
                self.episode_counts[key] = self.episode_counts.get(key, 0) + 1
                
        self.index = (self.index + 1) % self.n_steps
   
    def compute_accumulative_rewards(self):
        # Calculate average accumulative rewards of all episodes per agent
        if not self.episode_accumulative_rewards:  # If there are no accumulative rewards recorded
            return None
        
        total_accumulative_rewards = 0
        num_agents = 0
        for key, accumulative_rewards in self.episode_accumulative_rewards.items():
            if accumulative_rewards:  # Only consider agents with recorded rewards
                total_accumulative_rewards += accumulative_rewards
                num_agents += 1
        if num_agents == 0:  # Prevent division by zero
            return None
        
        avg_accumulative_rewards = total_accumulative_rewards / num_agents
        return avg_accumulative_rewards  # Return a single value representing the mean of the accumulative rewards of all agents
    
    def _update_best_reward(self, avg_reward):
        if avg_reward > self.best_reward:
            self.best_reward = avg_reward
            self.best_period = True 
        else: 
            self.best_period = False 

    def compute_average(self):
        valid_indices = np.where(self.counts > 0)
        valid_rewards = self.steps[valid_indices]
        valid_counts = self.counts[valid_indices]

        total_rewards = np.dot(valid_rewards, valid_counts)
        total_counts = np.sum(valid_counts)
        
        if total_counts == 0:
            return None

        avg_reward = total_rewards / total_counts
        self._update_best_reward(avg_reward)
        return avg_reward

    def get_episode_counts(self):
        """Returns the smallest number of episodes ended among all agents."""
        
        # Get all agents: those with ongoing episodes and those with ended episodes
        all_agents = set(self.accumulative_rewards.keys()).union(self.episode_counts.keys())
        
        # Find the smallest ended episode count among all agents
        min_ended_episodes = None
        for agent in all_agents:
            ended_episodes_for_agent = self.episode_counts.get(agent, 0)  # Consider 0 if the agent has no ended episodes
            if (min_ended_episodes is None) or (ended_episodes_for_agent < min_ended_episodes):
                min_ended_episodes = ended_episodes_for_agent

        return 0 if min_ended_episodes is None else min_ended_episodes
         