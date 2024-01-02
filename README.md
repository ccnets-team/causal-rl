# Weights & Biases CausalRL Benchmarks: OpenAI Gym Performance Analysis with Various Configurations (64 Batch Size & 100K Steps)
- https://wandb.ai/rl_tune/causal-rl-gym/
- https://wandb.ai/ccnets/rl-tune-gym/

# Table of Contents

- [Overview](#overview)
   * [Introduction](#introduction)
   * [Key Points](#key-points)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Features](#features)
- [Performance & Benchmarks](#performance--benchmarks)
  * [Algorithm Feature Checklist](#algorithm-feature-checklist)
- [API Documentation](#api-documentation)
- [Contribution Guidelines 🌟](#contribution-guidelines-)
- [Issue Reporting Policy 🐞](#issue-reporting-policy-)
- [Support & Contact](#support--contact)


# Overview

## **Introduction**

RL-Tune is a customizable Reinforcement Learning framework designed to offer flexibility, modularity, and adaptability to various RL scenarios. Below are detailed descriptions of the key features of the framework to help users leverage its full potential.

## **Key Points**

1. Tune RL Parameters for benchmarking, with a preset pipeline ready to execute reducing your effort in initial setups.
2. Develop flexible custom algorithms based on RL principles using role-based network, uniformly integrated with the latest RL frameworks.
3. Introduction of Causal RL integrating reverse-environment network into Actor-Critic framework learning the causal relationships between states, actions, and values, while maximizing accumulative rewards.

# ****Dependencies****

```python
conda create -name rl_tune python=3.9
conda activate rl_tune
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install mlagents==0.30
pip install protobuf==3.20
pip install gymnasium
pip install jupyter
pip install transformers==4.34.1
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
![1](https://github.com/ccnets-team/rl-tune/assets/75417171/4007a1c4-89c4-4727-bddb-4b332fb12bda)
![2](https://github.com/ccnets-team/rl-tune/assets/75417171/bfafafdd-b0f1-4210-866a-bcddfdb8953d)
![3](https://github.com/ccnets-team/rl-tune/assets/75417171/17494f60-ab20-4d93-9ca7-8ea44d6d8fd7)


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
![File (1)](https://github.com/ccnets-team/rl-tune/assets/75417171/e94d7922-1eba-4594-886b-428573bc19c4)

*HumanoidStandup-v4*


## Algorithm Feature Checklist
![File (2)](https://github.com/ccnets-team/rl-tune/assets/75417171/41cd231e-b13c-45a9-b21e-cb4a3ae3ed15)

# **API Documentation**

- We're currently in the process of building our official documentation webpage to better assist you. In the meantime, if you have any specific questions or need clarifications, feel free to reach out through our other support channels. We appreciate your patience and understanding!

# **Contribution Guidelines 🌟**
<details>
  <summary><strong> Click to see more </strong></summary>

  - We warmly welcome contributions from everyone! Here's how you can contribute:

![contribution process](https://github.com/ccnets-team/rl-tune/assets/66022264/34d55b31-5825-4e31-8407-690d00a4e502)

When you submit a Pull Request (PR) to our project, here's the process it goes through:

1. **Initial Check**: We first check if the PR is valid.
    - If not, it's rejected.
    - If valid, it proceeds to review.
2. **Review Process**:
    - If changes are needed, you'll receive feedback. Please make the necessary adjustments to your PR and resubmit. This review-feedback cycle may repeat until the PR is satisfactory.
    - If no changes are needed, the PR is approved.
3. **Testing**:
    - Approved PRs undergo testing.
    - If tests pass, your PR gets merged! 🎉
    - If tests fail, you'll receive feedback. Adjust your PR accordingly and it will go through the review process again.

Your contributions are invaluable to us. Please ensure you address feedback promptly to streamline the merge process.


# **Issue Reporting Policy 🐞**

Thank you for taking the time to report issues and provide feedback. This helps improve our project for everyone! To ensure that your issue is handled efficiently, please follow the guidelines below:

### **1. Choose the Right Template:**

We provide three issue templates to streamline the reporting process:

1. **Bug Report**: Use this template if you've found a bug or something isn't working as expected. Please provide as much detail as possible to help us reproduce and fix the bug.
2. **Feature Request**: If you have an idea for a new feature or think something could be improved, this is the template to use. Describe the feature, its benefits, and how you envision it.
3. **Custom Issue Template**: For all other issues or general feedback, use this template. Make sure to provide sufficient context and detail.

### **2. Search First:**

Before submitting a new issue, please search the existing issues to avoid duplicates. If you find a similar issue, you can add your information or 👍 the issue to show your support.

### **3. Be Clear and Concise:**

- **Title**: Use a descriptive title that summarizes the issue.
- **Description**: Provide as much detail as necessary, but try to be concise. If reporting a bug, include steps to reproduce, expected behavior, and actual behavior.
- **Screenshots**: If applicable, add screenshots to help explain the issue.

### **4. Use Labels:**

If possible, categorize your issue using the appropriate GitHub labels. This helps us prioritize and address issues faster.

### **5. Stay Engaged:**

After submitting an issue, please check back periodically. Maintainers or other contributors may ask for further information or provide updates.

Thank you for helping improve our project! Your feedback and contributions are invaluable.


</details>


# **Support & Contact**

Facing issues or have questions about our framework? We're here to help!

1. **Issue Tracker**:
    - If you've encountered a bug or have a feature request, please open an issue on our **[GitHub Issues page](https://github.com/ccnets-team/rl-tune/issues)**. Be sure to check existing issues to avoid duplicates.
2. **Social Media**:
    - Stay updated with announcements and news by following us on **[LinkedIn](https://www.linkedin.com/company/ccnets)**.
3. **Emergency Contact**:
    - If there are security concerns or critical issues, contact our emergency team at support@ccnets.org.

*Please be respectful and constructive in all interactions.*
