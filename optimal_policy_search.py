import numpy as np
from environment import MDP_GridSearch
from policy_evaluation import iterative_algorithm_policy_evaluation, calculate_G_t


def policy_iteration(mdp, gamma=0.9):
    """
    Perform policy iteration to find the optimal policy.
    :param mdp: MDP environment
    :return: Optimal policy and value function for each state

    compare pi_i and p_i+1
    V -> iterative algorithm
    Q(s,a) = R(s,a) + gamma * (sum over_s' P(s'|s, a) * V(s'))
    pi_{i+1}(s) = argmax_a Q(s,a)
    """
    new_policy = np.random.randint(
        0, len(mdp.action_space), size=len(mdp.state_space)
    )  # Random initial policy

    V = np.zeros(len(mdp.state_space))  # Initialize value function
    Q = np.zeros(
        (len(mdp.state_space), len(mdp.action_space))
    )  # Initialize action-value function

    old_policy = np.zeros_like(new_policy)

    while not np.array_equal(new_policy, old_policy):
        old_policy = np.copy(new_policy)

        mdp.policy = new_policy  # Set initial policy in MDP

        V = iterative_algorithm_policy_evaluation(
            mdp, gamma=0.9, theta=1e-10
        )  # It requires mdp knowledge

        for s in range(len(mdp.state_space)):
            for a in range(len(mdp.action_space)):
                Q[s, a] = mdp.rew[s, a] + gamma * np.sum(mdp.action_matrix[a][s, :] * V)

        new_policy = np.argmax(Q, axis=1)

    return new_policy, V


def value_iteration(mdp, gamma=0.9, theta=0.001):
    """
    Perform value iteration to find the optimal policy.
    :param mdp: MDP environment
    :param gamma: Discount factor
    :param theta: Threshold for convergence
    :return: Optimal policy and value function for each state
    """

    V = np.zeros(len(mdp.state_space))
    V_action = np.zeros((len(mdp.state_space), len(mdp.action_space)))
    V_prev = np.zeros_like(V)

    delta = 0
    new_policy = np.zeros(len(mdp.state_space), dtype=int)

    while True:
        delta = 0
        for s in range(len(mdp.state_space)):
            for a in range(len(mdp.action_space)):
                V_action[s, a] = mdp.rew[s, a] + gamma * np.sum(
                    mdp.action_matrix[a][s, :] * V_prev
                )

            V_prev[s] = V[s]
            V[s] = np.max(V_action[s, :])

        delta = max(delta, np.abs(V - V_prev).sum())
        if delta < theta:
            break

    for s in range(len(mdp.state_space)):
        best_action_value = -np.inf
        for a in range(len(mdp.action_space)):
            action_value = mdp.rew[s, a] + gamma * np.sum(
                mdp.action_matrix[a][s, :] * V
            )
            if action_value > best_action_value:
                best_action_value = action_value
                best_action = a

        new_policy[s] = best_action

    return new_policy, V


def eps_greedy_policy(mdp, Q, eps):
    """
    Generate an epsilon-greedy policy based on the given Q-values.

    Parameters:
        mdp: The Markov Decision Process environment.
        Q: The action-value function, a 2D numpy array of shape (num_states, num_actions).
        eps: The probability of choosing a random action (exploration rate).

    Returns:
        new_policy: A numpy array representing the policy, where each entry is the chosen action for a state.
    """
    best_q_policy = np.argmax(Q, axis=1)
    new_policy = np.zeros(len(mdp.state_space), dtype=int)

    for s in range(len(mdp.state_space)):
        if np.random.rand() < eps:
            feasible_actions = [
                mdp.map_action[a]
                for a in mdp.action_space
                if a != mdp.map_number_to_action[best_q_policy[s]]
            ]  # Get feasible actions
            if feasible_actions:
                new_policy[s] = np.random.choice(feasible_actions)
        else:
            new_policy[s] = best_q_policy[s]

    # After new_policy is computed (as a vector of actions for each state)
    policy_matrix = np.zeros((len(new_policy), len(mdp.action_space)), dtype=int)
    policy_matrix[np.arange(len(new_policy)), new_policy] = 1

    return policy_matrix, new_policy


def montecarlo_model_free_on_policy(mdp, steps, num_episodes, gamma=0.9):
    """
    Perform Monte Carlo model-free on-policy evaluation to estimate the optimal policy.
    Parameters:
        mdp: The Markov Decision Process environment.
        steps: Number of steps per episode.
        num_episodes: Total number of episodes to run.
        gamma: Discount factor for future rewards (default is 0.9).

    Returns:
        new_policy: A numpy array representing the estimated optimal policy for each state.
        Q: The action-value function as a 2D numpy array of shape (num_states, num_actions).

    Algorithm Details:
        - Uses first-visit Monte Carlo method to update Q-values.
        - Policy is improved using epsilon-greedy strategy after each episode.
        - The epsilon value decays over episodes to favor exploitation over exploration.
    """

    visited_states = set()
    N = np.zeros(
        (len(mdp.state_space), len(mdp.action_space))
    )  # State-action visit counts
    Q = np.zeros((len(mdp.state_space), len(mdp.action_space)))
    k = 0
    eps = 1
    new_policy_matrix, new_policy = eps_greedy_policy(mdp, Q, eps)

    for n in range(num_episodes):
        mdp.policy = new_policy_matrix  # Set current policy in MDP
        episode = mdp.generate_episode(steps)
        visited_states.clear()

        for i, (state, action, _, _) in enumerate(episode):
            if (state, action) not in visited_states:
                visited_states.add((state, action))
                N[state - 1, action] += 1
                G_t = calculate_G_t(episode[i:], gamma)
                Q[state - 1, action] = Q[state - 1, action] + 1 / N[
                    state - 1, action
                ] * (G_t - Q[state - 1, action])

        k = k + 0.1
        eps = 1 / np.sqrt(k)

        new_policy_matrix, new_policy = eps_greedy_policy(mdp, Q, eps)

    print(new_policy_matrix)
    return new_policy, Q


def td_model_free_on_policy(mdp, steps, num_episodes, alpha=0.2, gamma=0.9):
    """
    Perform Temporal Difference (TD) model-free on-policy evaluation to estimate the optimal policy.
    Parameters:
        mdp: The Markov Decision Process environment.
        steps: Maximum number of steps per episode.
        num_episodes: Number of episodes to run.
        gamma: Discount factor for future rewards (default is 0.9).
    Returns:
        best_policy: A numpy array representing the optimal action for each state.
        Q: The learned action-value function as a 2D numpy array (states x actions).
    """
    Q = np.zeros((len(mdp.state_space), len(mdp.action_space)))
    eps = 1  # initial exploration rate
    i = 0  # to count steps in episode
    k = 0  # to decay epsilon

    for episode_idx in range(num_episodes):
        s_t = mdp.starting_position
        done = False
        i = 0
        # Choose action according to current policy
        a_t_w = np.random.choice(mdp.action_space, p=mdp.policy[s_t - 1])
        a_t = mdp.map_action[a_t_w]

        while not done and i < steps:
            # Update policy after every step
            new_policy_matrix, new_policy = eps_greedy_policy(mdp, Q, eps)
            mdp.policy = new_policy_matrix

            # Take action, observe reward and next state
            r_t = mdp.rew[s_t - 1, a_t]
            next_state_prob_sum = mdp.action_matrix[a_t][s_t - 1].sum()
            if next_state_prob_sum == 0:
                s_t_1 = s_t
            else:
                s_t_1 = (
                    mdp.action_matrix[a_t][s_t - 1].argmax() + 1
                )  # TODO: stochastic transitions

            # Choose next action according to updated policy (for SARSA)
            a_t_1_w = np.random.choice(mdp.action_space, p=mdp.policy[s_t_1 - 1])
            a_t_1 = mdp.map_action[a_t_1_w]

            # TD update (SARSA)
            Q[s_t - 1, a_t] = Q[s_t - 1, a_t] + alpha * (
                r_t + gamma * Q[s_t_1 - 1, a_t_1] - Q[s_t - 1, a_t]
            )

            # Check for terminal state
            if s_t_1 == mdp.final_position:
                done = True

            # Prepare for next step
            s_t = s_t_1
            a_t = a_t_1
            i += 1
            k = k + 1

            # Decay epsilon after each episode
            eps = 1 / np.sqrt(k)

    # After training, return the greedy policy and Q
    best_policy = np.argmax(Q, axis=1)
    return best_policy, Q


def q_learning_model_free_off_policy(mdp, steps, num_episodes, alpha=0.2, gamma=0.9):
    """
    Perform Q-learning model-free off-policy evaluation to estimate the optimal policy.
    Parameters:
        mdp: The Markov Decision Process environment.
        steps: Maximum number of steps per episode.
        num_episodes: Number of episodes to run.
        gamma: Discount factor for future rewards (default is 0.9).
    Returns:
        best_policy: A numpy array representing the optimal action for each state.
    """
    Q = np.zeros((len(mdp.state_space), len(mdp.action_space)))
    eps = 1  # initial exploration rate
    i = 0  # to count steps in episode
    k = 0  # to decay epsilon

    for episode_idx in range(num_episodes):
        s_t = mdp.starting_position
        done = False
        i = 0

        while not done and i < steps:
            # Epsilon-greedy action selection (behavior policy)
            if np.random.rand() < eps:
                a_t = np.random.randint(0, len(mdp.action_space))
            else:
                a_t = np.argmax(Q[s_t - 1, :])

            # Take action, observe reward and next state
            r_t = mdp.rew[s_t - 1, a_t]
            next_state_prob_sum = mdp.action_matrix[a_t][s_t - 1].sum()
            if next_state_prob_sum == 0:
                s_t_1 = s_t
            else:
                s_t_1 = (
                    mdp.action_matrix[a_t][s_t - 1].argmax() + 1
                )  # TODO: stochastic transitions

            # TD update (Q-learning)
            Q[s_t - 1, a_t] = Q[s_t - 1, a_t] + alpha * (
                r_t + gamma * np.max(Q[s_t_1 - 1, :]) - Q[s_t - 1, a_t]
            )

            # Check for terminal state
            if s_t_1 == mdp.final_position:
                done = True

            # Prepare for next step
            s_t = s_t_1
            i += 1
            k = k + 1  # TODO: find a common decay for all algorithms

            # Decay epsilon after each episode
            eps = 1 / np.sqrt(k)

    # After training, return the greedy policy and Q
    best_policy = np.argmax(Q, axis=1)
    return best_policy, Q


if __name__ == "__main__":
    matrix = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])

    starting_position = 8
    final_position = 2

    mdp = MDP_GridSearch(matrix, starting_position, final_position, random_policy=False)

    best_pol, V = policy_iteration(mdp)
    print(f"Best policy, policy iteration: {best_pol}")
    print(f"V found: {V}")

    best_pol, V = value_iteration(mdp)
    print(f"Best policy, value iteration: {best_pol}")
    print(f"V found: {V}")

    steps = 20
    num_episodes = 5000
    best_pol, Q = montecarlo_model_free_on_policy(
        mdp, steps=steps, num_episodes=num_episodes
    )
    print(f"Best policy, MC iteration: {best_pol}")
    print(f"Q found: {Q}")

    steps = 20
    num_episodes = 5000
    best_pol, Q = td_model_free_on_policy(mdp, steps=steps, num_episodes=num_episodes)
    print(f"Best policy, TD iteration: {best_pol}")
    print(f"Q found: {Q}")

    steps = 20
    num_episodes = 5000
    best_pol, Q = q_learning_model_free_off_policy(
        mdp, steps=steps, num_episodes=num_episodes
    )
    print(f"Best policy, Q-Learning iteration: {best_pol}")
    print(f"Q found: {Q}")
