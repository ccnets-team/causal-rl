# Causal Reinforcement Learning Framework by CCNets

[![Static Badge](https://img.shields.io/badge/Release-v1.0.1-%25%2300FF00)](https://github.com/ccnets-team/causal-rl)
[![Static Badge](https://img.shields.io/badge/LICENSE-DUAL-%23512BD4)](./LICENSE/)
[![Static Badge](https://img.shields.io/badge/Python-3.9.18-%233776AB)](https://www.python.org/)
[![Static Badge](https://img.shields.io/badge/PyTorch-2.1.2-%23EE4C2C)](https://pytorch.org/get-started/locally/)
[![Static Badge](https://img.shields.io/badge/OpenAI%20Gym-0.29.1-%230081A5)](https://gymnasium.farama.org/environments/mujoco/)
[![Static Badge](https://img.shields.io/badge/Unity%20MLagents-0.30.0-%23000000)](https://github.com/Unity-Technologies/ml-agents)
[![Static Badge](https://img.shields.io/badge/%F0%9F%A4%97GPT%20model-Hugging%20Face-%23FF9D0B)](https://huggingface.co/gpt2)
[![Static Badge](https://img.shields.io/badge/CCNets-LinkedIn-%230A66C2)](https://www.linkedin.com/company/ccnets/)
[![Static Badge](https://img.shields.io/badge/Patent-Google-%234285F4)](https://patents.google.com/patent/WO2023167576A2/)


# Table of Contents

- [🎈 **Overview**](#🎈-overview)
   * [Introduction](#introduction)
   * [Key Points](#key-points)
- [❗️ **Dependencies**](#❗️-dependencies)
- [📥 **Installation**](#📥-installation)
- [🏃 **Quick Start**](#🏃-quick-start)
- [📖 **Features**](#📖-features)
    - [✔️ Algorithm Feature Checklist](#✔️-algorithm-feature-checklist)
    - [📗 Algorithms Implementation](#📗-algorithms-implementation)
    - [📈 CausalRL Benchmarks](#📈-causalrl-benchmarks)
        - [Download and Use Model(W&B)](#download-and-use-modelwb)
- [🔎 **API Documentation**](#🔎-api-documentation)
- [🌟 **Contribution Guidelines**](#🌟-contribution-guidelines-)
- [🐞 **Issue Reporting Policy**](#🐞-issue-reporting-policy-)
- [✉️ **Support & Contact**](#✉️-support--contact)




# 🎈 Overview

## **Introduction**

Causal RL is an innovative Reinforcement Learning framework that utilizes three networks: Actor, Critic, and Reverse Environment, to learn the causal relationships between states, actions, and values while maximizing cumulative rewards. This introduction provides detailed descriptions of the framework's key features to help users leverage the full potential of Causal RL.

## **Key Points**

1. Introduction of Causal RL integrating reverse-environment network into Actor-Critic framework learning the causal relationships between states, actions, and values, while maximizing accumulative rewards.

2. **Language Model Training with Reverse Causal Mask**: Causal RL utilizes a reverse mask during training to enhance its understanding of causal relationships between states and actions, leading to improved learning efficiency and strategic effectiveness.

3. **Efficient Parameter Tuning**: CausalRL offers a preset pipeline for parameter tuning for benchmarking, reducing the effort required in initial setups.

# ❗️ ****Dependencies****

```python
conda create -name crl python=3.9.18
conda activate crl
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install mlagents==0.30
pip install protobuf==3.20
pip install gymnasium==0.29.1
pip install mujoco==3.1.1
pip install jupyter
pip install transformers==4.34.1
```
# 📥 **Installation**

- Steps to install the framework.
- Note: Ensure you have the required dependencies installed as listed in the "Dependencies" section above.

**Installation Steps:**

1. Clone the repository:
    
    ```bash
    git clone https://github.com/ccnets-team/causal-rl.git
    ```
    
2. Navigate to the directory and install the required packages:
    
    ```bash
    cd causal-rl
    pip install -r requirements.txt
    ```

# 🏃 **Quick Start**

- A basic example to help users get up and running immediately.

### 1. Import Library
```python
from utils.setting.env_settings import analyze_env
import torch

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
```

### 2. Set Environment
```python
from utils.setting.env_settings import analyze_env

env_config, rl_params = analyze_env(env_name = "HumanoidStandup-v4")
```

### 3. Initializing and Running Causal RL Training Process

```python
from causal_rl import CausalRL

with CausalRL(env_config, rl_params, device, use_print = False, use_wandb = False) as causal_rl:
    causal_rl.train(resume_training = False, use_graphics = False) 
    causal_rl.test(max_episodes = 100, use_graphics = False)
    # Temporary setting of use_graphics = True is not supported due to recent `Mujoco` module update changes.
    
```

# 📖 **Features**

**1. Manageable RL Parameters**

CausalRL facilitates structured management of RL Parameters, allowing users to easily organize, store, and compare parameters, which provides more coherent configurations for diverse RL problems.

```python
from utils.setting.env_settings import analyze_env

env_config, rl_params = analyze_env(env_name = "HumanoidStandup-v4")

rl_params.algorithm.gpt_seq_length = 16
rl_params.normalization.state_normalizer = "running_mean_std"
```

**2. Enhancing CausalRL with GPT** 

```python
from nn.gpt import GPT

class NetworkParameters:
    def __init__(self, num_layers=5, d_model=256, dropout=0.01, 
                network_type=GPT):
        self.critic_network = network_type  
        self.actor_network = network_type  
        self.rev_env_network = network_type 
```

- Advanced Sequence Learning: GPT excels in processing sequence data, aiding agents in predicting future states and actions based on past events. This is particularly useful in strategy games like chess or Go.

- Complex Pattern Recognition: GPT's deep neural network structure is adept at learning and recognizing complex patterns, enabling agents to make informed decisions in intricate environments.

- Long-term Strategy Development: GPT's proficiency in learning long-term dependencies is crucial for strategic planning in fields like robotics or drones, focusing on goals like energy efficiency and safe navigation.

- **Reverse Causal Masking Benefits**: While GPT's actor and critic networks use forward masking, the reverse_env network employs reverse masking. This approach enhances the understanding and prediction of current states from past data, beneficial in complex scenarios like robotics for optimal decision-making.

<img src="https://github.com/ccnets-team/rl-tune/assets/95277008/93ae2640-8dc7-4665-972c-6b2c90557faf" style="max-width: 100%; height: auto;">


## ✔️ Algorithm Feature Checklist
<img src="https://github.com/ccnets-team/rl-tune/assets/95277008/2f9128a7-e428-4824-9423-725175d0c26e" style="max-width: 90%; height: auto;">


## 📗 Algorithms Implementation

| Algorithm | Implemantation | Docs |
|----------|----------|----------|
| **CausalRL**    | [causal_rl.py](https://github.com/ccnets-team/rl-tune/blob/all-gpt-causal-rl/training/trainer/causal_rl.py) | [Patent](https://patents.google.com/patent/WO2023167576A2/)     |
| **Advantage Actor-Critic (A2C)**    | [a2c.py](https://github.com/ccnets-team/rl-tune/blob/all-gpt-causal-rl/training/trainer/a2c.py) | [Docs](https://huggingface.co/blog/deep-rl-a2c)     |
| **Deep Deterministic Policy Gradient (DDPG)**    | [ddpg.py](https://github.com/ccnets-team/rl-tune/blob/all-gpt-causal-rl/training/trainer/ddpg.py) | [Docs](https://spinningup.openai.com/en/latest/algorithms/ddpg.html)     |
| **Deep Q-Network (DQN)**    | [dqn.py](https://github.com/ccnets-team/rl-tune/blob/all-gpt-causal-rl/training/trainer/dqn.py) | [Docs](https://huggingface.co/blog/deep-rl-a2c)     |
| **Soft Actor-Critic (SAC)**    | [sac.py](https://github.com/ccnets-team/rl-tune/blob/all-gpt-causal-rl/training/trainer/sac.py) | [Docs](https://spinningup.openai.com/en/latest/algorithms/sac.html)     |
| **Twin Delayed Deep Deterministic Policy Gradient (TD3)** | [td3.py](https://github.com/ccnets-team/rl-tune/blob/all-gpt-causal-rl/training/trainer/td3.py) | [Docs](https://spinningup.openai.com/en/latest/algorithms/td3.html)     |

<br></br>

# 📈 **CausalRL Benchmarks**
Discover the capabilities of CausalRL algorithms in various OpenAI Gym environments. Our benchmarks, adhering to optimized industrial requirements and running 100K steps with a 64 batch size, provide in-depth performance insights. Explore the detailed metrics:

- 📊 CausalRL Gym Benchmarks: W&B - https://wandb.ai/rl_tune/causal-rl-gym/
- 📜 CausalRL Reports and Insights: W&B - https://api.wandb.ai/links/ccnets/1334iuol


<img src="https://github.com/ccnets-team/rl-tune/assets/95277008/8f0d5f2d-ec93-4706-aa94-77b971a88ffb" style="max-width: 100%; height: auto;">


## Download and Use Model(W&B)

- Install the WandB Package
```bash
pip install wandb # If not installed
```

- Inintialize and Run WandB
```python
import wandb
run = wandb.init() # you may need to enter your API Key
```

- Download Artifact from WandB
```python
Artifact_Name = ... # including the specific model and version you want, and assign it
artifact = run.use_artifact('rl_tune/causal-rl-gym/{Artifact_Name}', type='model')
artifact_dir = artifact.download()
```

<br></br>

# 🔎 **API Documentation**

- We're currently in the process of building our official documentation webpage to better assist you. In the meantime, if you have any specific questions or need clarifications, feel free to reach out through our other support channels. We appreciate your patience and understanding!


# 🌟 **Contribution Guidelines**
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


# 🐞 **Issue Reporting Policy**

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


# ✉️ **Support & Contact**

Facing issues or have questions about our framework? We're here to help!

1. **Issue Tracker**:
    - If you've encountered a bug or have a feature request, please open an issue on our **[GitHub Issues page](https://github.com/ccnets-team/causal-rl/issues)**. Be sure to check existing issues to avoid duplicates.
2. **Social Media**:
    - Stay updated with announcements and news by following us on **[LinkedIn](https://www.linkedin.com/company/ccnets)**.
3. **Emergency Contact**:
    - If there are security concerns or critical issues, contact our emergency team at support@ccnets.org.

*Please be respectful and constructive in all interactions.*


# LICENSE
CAUSALRL is dual-licensed under the GNU General Public License version 3(GPLv3) and a separate Commercial License.

Please consult the [LICENSE](./LICENSE/) files in the repository for more detailed information on the licensing of CAUSALRL.
