# Overview

## **Introduction**

RL-Tune is a customizable Reinforcement Learning framework designed to offer flexibility, modularity, and adaptability to various RL scenarios. Below are detailed descriptions of the key features of the framework to help users leverage its full potential.

## **Key Points**

1. Tune RL Parameters for benchmarking, with a preset pipeline ready to execute reducing your effort in initial setups.
2. Develop flexible custom algorithms based on RL principles using role-based network, uniformly integrated with the latest RL frameworks.
3. Introduction of Causal RL integrating reverse-environment network into Actor-Critic framework learning the causal relationships between states, actions, and values, while maximizing accumulative rewards.

# ****Dependencies****

```python
conda create -name crl python=3.9
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install mlagents==0.30
pip install protobuf==3.20
pip install gymnasium
pip install jupyter
```
# **Installation**

- Steps to install the framework.
- Note: Ensure you have the required dependencies installed as listed in the "Dependencies" section above.

**Installation Steps:**

1. Clone the repository:
    
    ```bash
    git clone https://github.com/ccnets-team/rl-tune.git
    ```
    
2. Navigate to the directory and install the required packages:
    
    ```bash
    cd RL-Tune
    pip install -r requirements.txt
    ```

# **Quick Start**

- A basic example to help users get up and running immediately.

```python
env_config, rl_params = analyze_env(env_name = "PushBlock")

trainer = RLTrainer(rl_params, trainer_name='crl')  

rl_tune = RLTune(env_config, trainer, device, use_graphics = False, use_print = False)

rl_tune.train(on_policy=False)
```
![image](https://github.com/ccnets-team/rl-tune/assets/75417171/1531d697-6043-43c3-9888-d8d817ce73e8)
![1.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/c38b57b1-2243-423c-8306-07d332950ea1/4a9c44ca-ccf3-494e-8d64-928c9425daa4/1.png)
![2.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/c38b57b1-2243-423c-8306-07d332950ea1/240bd9f4-198c-442c-ac9c-a7362d50b2ab/2.png)
![3.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/c38b57b1-2243-423c-8306-07d332950ea1/23bc06c5-f9c6-449b-b598-6a44b1522dc0/3.png)

# Features

**1. Manageable RL Parameters**

RL-Tune facilitates structured management of RL Parameters, allowing users to easily organize, store, and compare parameters, which provides more coherent configurations for diverse RL problems.

```python
from utils.setting.env_settings import analyze_env

env_config, rl_params = analyze_env(env_name = "HumanoidStandup-v4")

rl_params.algorithm.num_td_steps = 5
rl_params.normalization.state_normalizer = "running_z_standardizer"
```

**2. Comprehensive RL Modules**

RL-Tune features comprehensive modules including memory, normalization, noise, exploration, and strategy managers, enabling a cohesive and seamless problem-solving experience in the RL domain.

**3. Flexible Network Role Assignments:**

RL-Tune’s flexible architecture facilitates distinct role assignments to different networks, optimizing the processes of development and management for various network configurations.

```python
from nn.roles.actor_network import SingleInputActor as QNetwork
from nn.roles.critic_network import SingleInputCritic
from nn.roles.reverse_env_network import RevEnv

q_network = QNetwork(network, env_config, network_params, exploration_params)
critic = SingleInputCritic(network, env_config, network_params)
reverse_env = RevEnv(network, env_config, network_params)

```

**4. Custom Algorithm Formulation with BaseTrainer Class:**

RL-Tune is built around the BaseTrainer class, empowering users to formulate custom algorithms and utilize commonalized functions for crafting powerful and efficient RL solutions.

```python
class BaseTrainer:
    def compute_values(self, states: torch.Tensor, rewards: torch.Tensor,
                       next_states: torch.Tensor, dones: torch.Tensor,
                       estimated_value: torch.Tensor) -> (torch.Tensor, torch.Tensor):
		...

class YourRLTrainer(BaseTrainer):
```

**5. Enhanced Algorithmic Components:**
    
- Numerical n-td steps:
Supports configurable numerical n-td steps for effective planning and learning.
    
- Expected Value Calculation:
Enables precise value predictions with the **`calculate_expected_value`** method.
    
- Value Loss Computation:
Optimizes value approximation with the **`calculate_value_loss`** method.
    
- Advantage Calculation:
Refines policy estimates with the **`calculate_advantage`** method.
    
- GAE Advantage:
Enhances policy optimization through the **`compute_gae_advantage`** method.
    
- Curiosity RL Integration:
Incorporates curiosity-driven RL components for enhanced exploration and learning in sparse reward environments.

# **Performance & Benchmarks**
![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/c38b57b1-2243-423c-8306-07d332950ea1/b461349a-615f-40ba-953c-899641a7cfb0/image.png)

*HumanoidStandup-v4*


## Algorithm Feature Checklist
![comparison.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/c38b57b1-2243-423c-8306-07d332950ea1/36664675-670c-4466-b155-42a9b51c60ce/comparison.png)

# **API Documentation**

- We're currently in the process of building our official documentation webpage to better assist you. In the meantime, if you have any specific questions or need clarifications, feel free to reach out through our other support channels. We appreciate your patience and understanding!

# **Contribution**

*We are actively seeking contributions to integrate this pioneering methodology with other frameworks, aiming to broaden its applicability and contribute to the advancement of reinforcement learning solutions.*

# **Support & Contact**

Facing issues or have questions about our framework? We're here to help!

1. **Issue Tracker**:
    - If you've encountered a bug or have a feature request, please open an issue on our **[GitHub Issues page](https://github.com/ccnets-team/rl-tune/issues)**. Be sure to check existing issues to avoid duplicates.
2. **Social Media**:
    - Stay updated with announcements and news by following us on **[LinkedIn](https://www.linkedin.com/company/ccnets)**.
3. **Emergency Contact**:
    - If there are security concerns or critical issues, contact our emergency team at support@ccnets.org.

*Please be respectful and constructive in all interactions.*
