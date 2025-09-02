# RL Basics: Policy, Evaluation, and Control

This repository implements fundamental reinforcement learning (RL) algorithms for Markov Decision Processes (MDPs) using a grid-based environment. The main components are policy representation, policy evaluation, and policy control techniques.

## Environment

- [`MDP_GridSearch`](environment.py): Represents the grid-based MDP environment. States and actions are encoded as integers and strings, respectively. The environment supports random and deterministic policies, computes reward matrices, and generates episodes for learning.

## Policy Representation

A **policy** defines the agent's behavior by mapping states to actions. In this workspace:
- Policies can be random (stochastic) or deterministic.
- Policies are represented as vectors (deterministic) or matrices (stochastic, with action probabilities per state).

## Policy Evaluation

Policy evaluation estimates the value function for a given policy, i.e., the expected return from each state when following the policy.

Implemented techniques:
- **Iterative Policy Evaluation**: [`iterative_algorithm_policy_evaluation`](policy_evaluation.py) computes state values by iteratively applying the Bellman expectation equation until convergence.
- **Monte Carlo Policy Evaluation**: [`montecarlo_policy_evaluation`](policy_evaluation.py) uses sampled episodes to estimate state values via first-visit returns.
- **Temporal Difference (TD) Policy Evaluation**: [`temporal_difference_policy_evaluation`](policy_evaluation.py) updates state values using bootstrapped estimates from sampled transitions.

## Policy Control

Policy control aims to find the optimal policy that maximizes expected returns.

Implemented techniques:
- **Policy Iteration**: [`policy_iteration`](optimal_policy_search.py) alternates between policy evaluation and policy improvement until convergence.
- **Value Iteration**: [`value_iteration`](optimal_policy_search.py) iteratively updates state values and derives the optimal policy.
- **Epsilon-Greedy Policy**: [`eps_greedy_policy`](optimal_policy_search.py) balances exploration and exploitation by selecting the best action with probability `1-epsilon` and a random action otherwise.
- **Monte Carlo Model-Free Control**: [`montecarlo_model_free_on_policy`](optimal_policy_search.py) uses first-visit MC updates and epsilon-greedy improvement to learn optimal policies without a model.
- **TD Model-Free Control (SARSA)**: [`td_model_free_on_policy`](optimal_policy_search.py) learns policies using TD updates and on-policy epsilon-greedy exploration.
- **Q-Learning Model-Free Control**: [`q_learning_model_free_off_policy`](optimal_policy_search.py) learns optimal policies using off-policy TD updates and epsilon-greedy exploration.
- **Deep Q-Network (DQN)**: [`dqn`](value_function_approximator.py) uses neural networks to approximate Q-values and optimize policies via experience replay and target networks.

## Usage

Run any of the main scripts to see the algorithms in action. For example:

```sh
python optimal_policy_search.py
```

This will print the results of policy iteration, value iteration, Monte Carlo, TD, and Q-learning algorithms.

## Requirements

See [requirements.txt](requirements.txt) for dependencies.

## File Overview

- [environment.py](environment.py): MDP environment and episode generation.
- [policy_evaluation.py](policy_evaluation.py): Policy evaluation algorithms.
- [optimal_policy_search.py](optimal_policy_search.py): Policy control algorithms.
- [value_function_approximator.py](value_function_approximator.py): DQN implementation.

## References

- Sutton & Barto, "Reinforcement Learning: An Introduction"