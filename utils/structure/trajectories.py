import numpy as np

class BatchTrajectory:
    def __init__(self, state, action, reward, next_state, done, buffer_indices = None, cumulative_sizes = None):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done
        self.buffer_indices = buffer_indices
        self.cumulative_sizes = cumulative_sizes
        self.td_errors = None
        self.padding_mask = None
    
    def push_td_errors(self, td_errors, padding_mask):
        self.td_errors = td_errors
        self.padding_mask = padding_mask

    def __iter__(self):
        yield self.state
        yield self.action
        yield self.reward
        yield self.next_state
        yield self.done
        
class MultiEnvTrajectories:
    def __init__(self, env_ids=None, agent_ids=None, states=None, actions=None, rewards=None, next_states=None, dones_terminated=None, dones_truncated=None, values=None):
        self.env_ids = env_ids 
        self.agent_ids = agent_ids 
        self.states = states 
        self.actions = actions 
        self.rewards = rewards 
        self.next_states = next_states 
        self.dones_terminated = dones_terminated 
        self.dones_truncated = dones_truncated 
        self.values = values 

    def _add_attribute(self, attr_name, val):
        attr = getattr(self, attr_name)
        # If values is not None and is non-empty, extend the attribute
        if val is not None and len(val) > 0:
            # Convert values to numpy array if it's a list
            if isinstance(val, list):
                val = np.array(val)
            # If the attribute is None, set it to values
            if attr is None or len(attr) == 0:
                setattr(self, attr_name, val)
            else:
                # Convert the attribute to numpy array if it's a list
                if isinstance(attr, list):
                    attr = np.array(attr)
                # concatenate existing attribute and new values
                setattr(self, attr_name, np.concatenate((attr, val), axis=0))

    def add(self, env_ids, agent_ids, states, actions, rewards, next_states, dones_terminated, dones_truncated, values=None):
        self._add_attribute('env_ids', env_ids)
        self._add_attribute('agent_ids', agent_ids)
        self._add_attribute('states', states)
        self._add_attribute('actions', actions)
        self._add_attribute('rewards', rewards)
        self._add_attribute('next_states', next_states)
        self._add_attribute('dones_terminated', dones_terminated)
        self._add_attribute('dones_truncated', dones_truncated)
        self._add_attribute('values', values)

    