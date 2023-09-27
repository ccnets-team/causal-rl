# CRL
HOW TO USE INTERFACE!!

1. there are only 2 environments available: UNITY or OPEN AI GYM

2. write the name of your environment example: (env_name = "Humanoid-v4")

        * after inputing your environment, the best parameters for the training will be set.

3. after that you have to select one of the 7 trainers: trainer = RLTrainer(trainer_type='********', device=device)

            *a2c - the Advantage Actor Critic (A2C) algorithm combines two types of Reinforcement Learning algorithms (Policy Based and Value Based) together.

            *crl - causal reinforcement learning using critic actor and reverse environment.

            *ddpg - Deep Deterministic Policy Gradient (DDPG) is an algorithm which concurrently learns a Q-function and a policy. It uses off-policy data and the      Bellman equation to learn the Q-function, and uses the Q-function to learn the policy.

            *dqn - Deep Q-Network, is a model used in reinforcement learning that combines the traditional Q-learning algorithm with deep neural networks. 

            *ppo - Proximal Policy Optimization (PPO)  algorithms are policy gradient methods, which means that they search the space of policies rather than assigning values to state-action pairs.

            *sac - Soft Actor Critic (SAC) is an algorithm that optimizes a stochastic policy in an off-policy way, forming a bridge between stochastic policy optimization and DDPG-style approaches. 

            *td3 - The twin-delayed deep deterministic policy gradient (TD3) algorithm is a model-free, online, off-policy reinforcement learning method. A TD3 agent is an actor-critic reinforcement learning agent that searches for an optimal policy that maximizes the expected cumulative long-term reward.

4. Choose if you want to see a visualization of training -  no_graphics = False/True

5. Run your project


if you have further questions or would like to receive help do not hesitate to reach out to us.

