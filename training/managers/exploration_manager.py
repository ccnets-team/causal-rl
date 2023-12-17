def compute_lin_decay_factor(initial_exploration, min_exploration, max_steps, decay_percentage):
    decay_steps = decay_percentage * max_steps
    return (min_exploration - initial_exploration) / decay_steps

def compute_exp_decay_factor(initial_exploration, min_exploration, max_steps, decay_percentage):
    decay_steps = decay_percentage * max_steps
    return (min_exploration / initial_exploration) ** (1/decay_steps)

class ExplorationUtils:
    def __init__(self, exploration):
        self.initial_exploration = exploration.initial_exploration
        self.min_exploration = exploration.min_exploration
        self.decay_percentage = exploration.decay_percentage
        self.max_steps = exploration.max_steps

        if hasattr(exploration, "decay_mode"):
            self.decay_mode = exploration.decay_mode
        else:
            self.decay_mode = "linear"  # default value

        if self.decay_mode == "linear":
            self.decay_factor = compute_lin_decay_factor(self.initial_exploration, self.min_exploration, self.max_steps, self.decay_percentage)
        elif self.decay_mode == "exponential":
            self.decay_factor = compute_exp_decay_factor(self.initial_exploration, self.min_exploration, self.max_steps, self.decay_percentage)

        self.exploration_rate = self.initial_exploration
        
    def update_exploration_rate(self):
        if self.decay_mode == "linear":
            self.exploration_rate = max(self.exploration_rate + self.decay_factor, self.min_exploration)
        elif self.decay_mode == "exponential":
            self.exploration_rate = max(self.decay_factor * self.exploration_rate, self.min_exploration)

    def get_exploration_rate(self):
        return self.exploration_rate