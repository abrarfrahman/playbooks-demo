# Playbooks for Collaborative Intelligence
***Investigating Misaligned Behaviors in Multi-Agent Systems Using Sports Analytics***

[preprint](https://github.com/abrarfrahman/playbooks-demo/blob/main/%5Bsafety-preprint%5D-Playbooks-Collaborative-Intelligence.pdf) | [colab notebook](https://colab.research.google.com/drive/1uzJJUTXbxCMPTJZoFbN4H3h0A1prL83M#scrollTo=Z2wuyS3kWCTf) | [hackathon slides](https://docs.google.com/presentation/d/1J57zn_vpVaRQ7B5yWqAn4_9LpjvPb0drGMuaYHLetoA/edit?usp=sharing)


Authors: Abrar Rahman, Anish Sundar

<img src="https://github.com/matteobettini/vmas-media/raw/main/media/scenarios/football.gif?raw=true" height="200"> <img src="https://pettingzoo.farama.org/_images/mpe_simple_tag.gif" width="200">

## AI-SAF: The AI Sports Analytics Framework

A novel multi-agent interpretability framework. Sports metaphors offer a powerful lens to interpret chaotic, multi-agent systems where individual roles, team dynamics, and external variables shape outcomes. This project investigates AI safety in multi-agent systems, with a focus on interpretability and alignment challenges. Inspired by how “ball-hog” behavior can harm team success despite impressive individual stats. By studying emergent patterns and coordination trade-offs, this framework aims to improve the reliability and safety of multi-agent AI systems.

### Value Over Replacement Agent (VORA)
**Definition:** Measures how much better (or worse) an agent or human performs compared to a baseline "replacement agent."

| **Formula** |
|-------------|
| $\text{VORA} = \text{Agent Performance} - \text{Baseline Performance}$ |

### Team Playmaking (TPM, or “assists”)
**Definition:** Measures how effectively an agent enables downstream success for humans or other agents.

| **Formula** |
|-------------|
| $\text{Assists(a)} = \text{Sum or Number of Meaningful Outputs Used Downstream}$ |

### Task Oversights (TO, or “turnovers”)
**Definition:** Quantifies the frequency and severity of mistakes introduced by an agent or human.

| **Formula** |
|-------------|
| $\text{TO} = \text{Sum or Number of Serious Errors Introduced by Agent}$ |

### Error Reduction and Recovery (ERR, or “rebounds”)
**Definition:** Measures an agent’s ability to recover from or correct errors introduced by others.

| **Formula** |
|-------------|
| $\text{ERR} = \text{Sum or Number of Errors Corrected}$ |

### Usage Rate (UR, or “offensive load”)
**Definition:** Proportion of the team’s workload handled by an agent or human. Default cost function is the number of tasks attempted.

| **Formula** |
|-------------|
| $\text{UR} = \frac{\text{Tasks Handled by Agent}}{\text{Cost Function}}$ |

***Note:*** UR can be extended to other cost functions, depending on which constraint to optimize for (ex: dollars, throughput ← TPS, latency ← TTFT, user engagement etc)

### Derived Efficiency Metrics

<img width="288" alt="Screenshot 2024-12-19 at 1 45 54 PM" src="https://github.com/user-attachments/assets/4bff6940-0116-475d-95f2-54e06f05037f" />

[from basketball reference](https://www.basketball-reference.com/leaders/per_career.html)

| **Metric**                     | **Common Name**   | **Definition**                                                       | **Formula**                                                                 |
|---------------------------------|-------------------|-----------------------------------------------------------------------|-----------------------------------------------------------------------------|
| **Efficiency-Adjusted Team Playmaking** | Assist Rate      | Normalizes the assist score based on the agent’s Usage Rate.               | $\text{E-TPM} = \frac{\text{TPM}}{\text{UR}}$                             |
| **Efficiency-Adjusted Task Oversights** | Turnover Ratio   | Normalizes turnover rates by Usage Rate.                                  | $\text{E-TO} = \frac{\text{TO}}{\text{UR}}$                               |
| **Efficiency-Adjusted Error Reduction and Recovery** | Rebound Rate     | Evaluates how effectively an agent corrects errors relative to its Usage Rate. | $\text{E-ERR} = \frac{\text{ERR}}{\text{UR}}$                             |


## Proof of Concept

This repository explores the application of Collaborative Intelligence to Multi-Agent Systems (MAS), focusing on misaligned behaviors through the lens of sports analytics. The goal is to understand the interaction dynamics in multi-agent environments and how these interactions can be modeled, trained, and analyzed using Deep Reinforcement Learning (DRL) techniques. We specifically focus on a football environment from the VMAS simulator, which is part of the MultiAgentParticleEnvironments (MPE) suite. For those new to the concept, MPE environments are designed to enable simulations of multi-agent systems and allow experimentation with various Multi-Agent Reinforcement Learning (MARL) algorithms.

### Enviornment Setup

We use the `football` environment from VMAS, a vectorized implementation of multi-agent systems that can run on GPUs. This provides a massive speedup compared to the traditional PettingZoo environments, which operate on CPUs. The football simulation is available only on VMAS, but sample code is included to run a simpler PettingZoo environment (e.g., `simple_tag_v3`) as a proof of concept.

Key Components:
- [PettingZoo](https://pettingzoo.farama.org/index.html): A framework for multi-agent environments on CPUs.
- [VMAS](https://github.com/proroklab/VectorizedMultiAgentSimulator): A PyTorch-based library for vectorized MPE environments, optimized for GPUs.

### Algorithmic Approach

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

### Installation Instructions

To set up the environment for this notebook, please install the following dependencies:

```bash
!pip3 install torchrl
!pip3 install vmas
!pip3 install pettingzoo[mpe]==1.24.3
!pip3 install tqdm
!pip3 install av
!apt-get install python3-opengl
```


Submitted to the AI Safety track for the RDI Agent MOOC Hackathon, Dec 2024

<img src="https://rdi.berkeley.edu/llm-agents-hackathon/assets/img/Berkeley_RDI_Logo.png" alt="Berkeley RDI" title="Berkeley RDI" width="150">
