import numpy as np
from environment import MDP_GridSearch


def iterative_algorithm_policy_evaluation(mdp, gamma=0.9, theta=1e-10):
    """
    Perform policy evaluation using the iterative algorithm.
    :param mdp: MDP environment
    :param gamma: Discount factor
    :param theta: Threshold for convergence
    :return: Value function for each state

    V_{k}(s) = R(s, pi(s)) + gamma * sum_over_s' P(s'|s, pi(s)) * V_{k-1}(s') for deterministic case
    V_{k}(s) = sum_over_a pi(a | s)[R(s, a) + gamma * sum_over_s' P_pi(s'|s, a) * V_{k-1}(s')] for stochastic case
    """
    V = np.zeros(len(mdp.state_space))  # Initialize value function

    while True:
        delta = 0
        V_prev = np.copy(V)

        for s in range(len(mdp.state_space)):
            # if mdp.random_policy is False:
            future_action = mdp.policy[s]  # define the action

            action_matrix = mdp.action_matrix[
                future_action
            ]  # consider the action and retrieve the matrix for this action

            v = mdp.rew[s, future_action] + gamma * np.sum(action_matrix[s, :] * V_prev)

            """
            # for now the stochastic policy is not implemented for iterative algorithm, at the beginning of policy evaluation is taken the argmax action
             else:
                v = 0

                for a in mdp.action_matrix.keys():
                    v += mdp.policy[s, a] * (
                        mdp.rew[s, a]
                        + gamma * np.sum(mdp.action_matrix[a][s, :] * V_prev)
                    )

            """

            # Update delta for convergence check
            delta = max(delta, np.abs(v - V_prev[s]))
            V[s] = v

        # Check for convergence
        if delta < theta:
            break

    return V


# require episodic setting
def montecarlo_policy_evaluation(mdp, steps=10, num_episodes=100, gamma=0.9):
    """
    Perform policy evaluation using Monte Carlo method.
    :param mdp: MDP environment
    :param num_episodes: Number of episodes to sample
    :param gamma: Discount factor
    :return: Value function for each state
    """
    V = np.zeros(len(mdp.state_space))
    N = np.zeros(len(mdp.state_space))  # State visit counts
    G = np.zeros(len(mdp.state_space))  # Cumulative returns

    for i in range(num_episodes):
        episode = mdp.generate_episode(steps)
        visited_states = set()

        for t, (state, action, reward, next_state) in enumerate(episode):
            # first visit MC
            if state not in visited_states:
                visited_states.add(state)
                N[state - 1] += 1
                G[state - 1] += calculate_G_t(episode[t:], gamma)
                V[state - 1] = G[state - 1] / N[state - 1]

    return V


def calculate_G_t(episode, gamma):
    G_t = 0
    for t in range(len(episode)):
        state, action, reward, next_state = episode[t]
        G_t += (gamma**t) * reward

    return G_t


def temporal_difference_policy_evaluation(
    mdp, steps, num_episodes, gamma=0.9, alpha=0.1
):
    """
    Perform policy evaluation using Temporal Difference method.
    :param mdp: MDP environment
    :param gamma: Discount factor
    :param alpha: Learning rate
    :param steps: Number of steps in each episode
    :param num_episodes: Number of episodes to sample
    :return: Value function for each state

    V(s) = V(s) + alpha*(r + gamma * V(s+1) - V(s))
    """
    V = np.zeros(len(mdp.state_space))  # Initialize value function
    for i in range(num_episodes):
        episode = mdp.generate_episode(
            steps
        )  # it does not require to be an episode, but a sequence of steps
        for t, (state, action, reward, next_state) in enumerate(episode):
            V[state - 1] = V[state - 1] + alpha * (
                reward + gamma * V[next_state - 1] - V[state - 1]
            )

    return V


if __name__ == "__main__":
    matrix = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])

    starting_position = 8
    final_position = 2

    mdp = MDP_GridSearch(matrix, starting_position, final_position)

    value_state = iterative_algorithm_policy_evaluation(mdp)
    print("Value State using Iterative Algorithm:")
    print(value_state)

    steps = 20
    num_episodes = 100
    value_state_MC = montecarlo_policy_evaluation(
        mdp, steps=steps, num_episodes=num_episodes
    )
    print(f"Monte Carlo Value State with {steps} steps and {num_episodes} episodes:")
    print(value_state_MC)

    steps = 20
    num_episodes = 1000
    value_state_MC = montecarlo_policy_evaluation(
        mdp, steps=steps, num_episodes=num_episodes
    )
    print(f"Monte Carlo Value State with {steps} steps and {num_episodes} episodes:")
    print(value_state_MC)

    steps = 20
    num_episodes = 2000
    value_state_TD = temporal_difference_policy_evaluation(
        mdp, steps=steps, num_episodes=num_episodes
    )
    print(f"TD Value State with {steps} steps and {num_episodes} episodes:")
    print(value_state_TD)
