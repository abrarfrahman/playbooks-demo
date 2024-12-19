# playbooks-demo

Authors: Abrar Rahman, Anish Sundar

[[preprint draft](https://github.com/abrarfrahman/playbooks-demo/blob/main/%5Bsafety-preprint%5D-Playbooks-Collaborative-Intelligence.pdf)]

This repository explores the application of Collaborative Intelligence to Multi-Agent Systems (MAS), focusing on misaligned behaviors through the lens of sports analytics. The goal is to understand the interaction dynamics in multi-agent environments and how these interactions can be modeled, trained, and analyzed using Deep Reinforcement Learning (DRL) techniques.

We specifically focus on a football environment from the VMAS simulator, which is part of the MultiAgentParticleEnvironments (MPE) suite. For those new to the concept, MPE environments are designed to enable simulations of multi-agent systems and allow experimentation with various Multi-Agent Reinforcement Learning (MARL) algorithms.

See also
- BenchMARL library: Provides state-of-the-art implementations of MARL algorithms using TorchRL.

## Environment Setup

We use the football environment from VMAS, a vectorized implementation of multi-agent systems that can run on GPUs. This provides a massive speedup compared to the traditional PettingZoo environments, which operate on CPUs. The football simulation is available only on VMAS, but sample code is included to run a simpler PettingZoo environment (e.g., simple_tag_v3) as a proof of concept.

Key Components:
-	PettingZoo: A framework for multi-agent environments on CPUs.
- VMAS: A PyTorch-based library for vectorized MPE environments, optimized for GPUs.

## Algorithmic Approach

In this project, we focus on Deep Deterministic Policy Gradient (DDPG), an off-policy actor-critic reinforcement learning algorithm that optimizes a deterministic policy based on gradients from the critic network. For multi-agent systems, we extend this approach to MADDPG (Multi-Agent DDPG), which allows multiple agents to learn in a cooperative-competitive environment.

The key difference in multi-agent settings is the need for decentralized execution, where each agent has its own policy and decision-making process based solely on its local observation. However, the critic can be either centralized (global information is shared among agents) or decentralized (only local information is used), depending on the algorithm.

Centralized vs. Decentralized Critics
- MADDPG: The critic takes the global state and global action as input, meaning all agent information is shared and the training is centralized.
- IDDPG: The critic uses only local observations and actions, supporting decentralized training, where each agent operates independently.

Notebook Structure
	1.	Hyperparameter Setup: Define hyperparameters to control the training environment and agent behavior.
	2.	Environment Construction: Build the multi-agent environment using TorchRL’s wrapper for either PettingZoo or VMAS.
	3.	Policy & Critic Networks: Create the actor-critic networks, exploring the trade-offs in parameter sharing and critic centralization.
	4.	Sampling & Replay Buffer: Set up a replay buffer for storing agent interactions and sampling for training.
	5.	Simulation & Metrics: Aggregate simulation data and compute team-level and agent-level metrics, simulating a “box score.”
	6.	Visualization: Render the environment and visualize the agent’s learned policy, before and after training, if running on a machine with a GUI.

Installation Instructions

To set up the environment for this notebook, please install the following dependencies:

```bash
!pip3 install torchrl
!pip3 install vmas
!pip3 install pettingzoo[mpe]==1.24.3
!pip3 install tqdm
!pip3 install av
!apt-get install python3-opengl
```

Further Reading
	•	Deep Deterministic Policy Gradients: The original DDPG paper (Lillicrap et al., 2015).
	•	Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments: The MADDPG paper.
	•	Reinforcement Learning: An Introduction (Sutton & Barto, 2018): The classic textbook for reinforcement learning.

License

This project is licensed under the MIT License - see the LICENSE file for details.
