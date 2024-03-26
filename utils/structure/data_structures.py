import torch

class BatchTrajectory:
    def __init__(self, trajectory_states, action, reward, done):
        self.trajectory_states = trajectory_states
        self.action = action
        self.reward = reward
        self.done = done

    def __iter__(self):
        yield self.trajectory_states
        yield self.action
        yield self.reward
        yield self.done
        
class AgentTransitions:
    def __init__(self, env_ids=None, agent_ids=None, states=None, actions=None, rewards=None, next_states=None, 
                 dones_terminated=None, dones_truncated=None, content_length=None):
        self.env_ids = env_ids 
        self.agent_ids = agent_ids 
        self.states = states 
        self.actions = actions 
        self.rewards = rewards 
        self.next_states = next_states 
        self.dones_terminated = dones_terminated 
        self.dones_truncated = dones_truncated 
        self.content_length = content_length 

    def _add_attribute(self, attr_name, val, dtype):
        attr = getattr(self, attr_name)
        # If values is not None and is non-empty, extend the attribute
        if val is not None and len(val) > 0:
            # Convert values to numpy array if it's a list
            if isinstance(val, list):
                val = torch.tensor(val, dtype = dtype)
            # If the attribute is None, set it to values
            if attr is None or len(attr) == 0:
                setattr(self, attr_name, val)
            else:
                # Convert the attribute to numpy array if it's a list
                if isinstance(attr, list):
                    attr = torch.tensor(attr, dtype = dtype)
                # concatenate existing attribute and new values
                setattr(self, attr_name, torch.concat((attr, val), dim=0))

    def add(self, env_ids, agent_ids, states, actions, rewards, next_states, dones_terminated, dones_truncated, content_length):
        self._add_attribute('env_ids', env_ids, dtype=torch.int)
        self._add_attribute('agent_ids', agent_ids, dtype=torch.int)
        self._add_attribute('states', states, dtype=torch.float)
        self._add_attribute('actions', actions, dtype=torch.float)
        self._add_attribute('rewards', rewards, dtype=torch.float)
        self._add_attribute('next_states', next_states, dtype=torch.float)
        self._add_attribute('dones_terminated', dones_terminated, dtype=torch.bool)
        self._add_attribute('dones_truncated', dones_truncated, dtype=torch.bool)
        self._add_attribute('content_length', content_length, dtype=torch.int)

    