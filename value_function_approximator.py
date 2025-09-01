import numpy as np
from environment import MDP_GridSearch
import torch.nn as nn
import torch


class DQNet(nn.Module):
    def __init__(self, states, actions, hidden_size=64):
        super().__init__()
        self.n_observation = len(states)
        self.n_actions = len(actions)
        self.fc1 = nn.Linear(self.n_observation, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.ReLU()
        self.output = nn.Linear(hidden_size, self.n_actions)

    def forward(self, x):
        """
        x: (state, action)
        y: G_t
        """
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        out = self.activation(out)
        out = self.output(out)

        return out


def greedy_dqn_step(mdp, s_t, policy_net, eps):
    """
    Take a step in the environment using an epsilon-greedy policy derived from the DQN.
    :param mdp: The MDP environment.
    :param s_t: Current state.
    :param policy_net: The current policy network.
    :param eps: Exploration factor (epsilon).
    :return: Tuple of (current state, action taken, reward received, next state).
    """
    if np.random.rand() < eps:
        a_t = np.random.randint(0, len(mdp.action_space))
    else:
        pred = policy_net(
            torch.nn.functional.one_hot(
                torch.tensor(s_t - 1), num_classes=len(mdp.state_space)
            ).float()
        )
        a_t = np.argmax(pred.detach().numpy())

    r_t = mdp.rew[s_t - 1, a_t]  # r(t)

    next_state_prob_sum = mdp.action_matrix[a_t][s_t - 1].sum()
    if next_state_prob_sum == 0:
        s_t_1 = s_t
    else:
        s_t_1 = (
            mdp.action_matrix[a_t][s_t - 1].argmax() + 1
        )  # TODO: stochastic transitions

    return s_t, a_t, r_t, s_t_1


def create_batch(target_net, mdp, buffer_sarsa, gamma, batch_size=32):
    """
    Create a batch of experiences for training the DQN.
    :param policy_net: The current policy network.
    :param mdp: The MDP environment.
    :param buffer_sarsa: The experience replay buffer containing (state, action, reward, next_state) tuples.
    :param gamma: Discount factor.
    :param batch_size: Number of experiences to sample for the batch.
    :return: Tensors for batch states, actions, and target Q-values.
    """
    batch_sarsa_i = np.random.choice(len(buffer_sarsa), batch_size)
    batch_sarsa = [buffer_sarsa[i] for i in batch_sarsa_i]
    batch_target = []
    batch_state = []
    batch_action = []

    for s_t, a_t_num, r_t, s_t_1 in batch_sarsa:
        if s_t_1 == mdp.final_position:
            target = r_t
        else:
            s_t_1_torch = torch.nn.functional.one_hot(
                torch.tensor(s_t_1 - 1), num_classes=len(mdp.state_space)
            ).float()
            with torch.no_grad():
                pred = target_net(s_t_1_torch)
                target = r_t + gamma * pred.max().item()

        batch_target.append(torch.tensor([target], dtype=torch.float))
        batch_state.append(
            torch.nn.functional.one_hot(
                torch.tensor(s_t - 1), num_classes=len(mdp.state_space)
            ).float()
        )
        batch_action.append(torch.tensor([a_t_num], dtype=torch.int64))

    batch_target = torch.cat(batch_target, dim=0)
    batch_state = torch.stack(batch_state, dim=0)
    batch_action = torch.cat(batch_action, dim=0)
    return batch_state, batch_action, batch_target


def optimize_dqn(policy_net, batch_state, batch_action, batch_target):
    mse = nn.MSELoss()
    lr = 0.001
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)

    optimizer.zero_grad()
    # prediction
    pred = policy_net(batch_state)
    pred_act = pred.gather(1, batch_action.unsqueeze(1)).squeeze(1)
    loss = mse(pred_act, batch_target)
    loss.backward()
    # optimizer step
    optimizer.step()

    return policy_net


def dqn(mdp, steps, episodes, gamma=0.9):
    """
    Deep Q-Network (DQN) algorithm for policy optimization.
    :param mdp: MDP environment
    :param steps: Number of steps in each episode
    :param episodes: Number of episodes to sample
    :param gamma: Discount factor
    :return: Q-value function for each state-action pair
    """
    eps = 1  # exploration factor
    k = 0  # decay factor
    buffer_sarsa = []  # experience replay buffer
    batch_size = 8  # batch size for experience replay
    C = 3  # target network update frequency

    policy_net = DQNet(states=mdp.state_space, actions=mdp.action_space)
    target_net = DQNet(states=mdp.state_space, actions=mdp.action_space)

    for num_episode in range(episodes):
        i = 0
        s_t = mdp.starting_position  # s(t)
        print(f"Episode {num_episode + 1}/{episodes}")
        while i < steps:
            # Take action, observe reward and next state
            s_t, a_t, r_t, s_t_1 = greedy_dqn_step(mdp, s_t, policy_net, eps)
            buffer_sarsa.append((s_t, a_t, r_t, s_t_1))

            # Optimizing DQN
            if len(buffer_sarsa) >= batch_size:
                batch_state, batch_action, batch_target = create_batch(
                    target_net, mdp, buffer_sarsa, gamma, batch_size=batch_size
                )
                policy_net_params_improved = optimize_dqn(
                    policy_net, batch_state, batch_action, batch_target
                )
                policy_net.load_state_dict(policy_net_params_improved.state_dict())

            if s_t_1 == mdp.final_position:
                break
            s_t = s_t_1
            i += 1
            k += 0.5
            eps = 1 / np.sqrt(k)

            if k % C == 0:
                target_net.load_state_dict(policy_net.state_dict())

    # Derive policy from the trained DQN
    best_policy = np.zeros(len(mdp.state_space), dtype=int)
    for s in range(0, len(mdp.state_space)):
        s_t_torch = torch.nn.functional.one_hot(
            torch.tensor(s), num_classes=len(mdp.state_space)
        ).float()
        pred = policy_net(s_t_torch).detach().numpy()
        best_policy[s] = np.argmax(pred)

    return pred, best_policy


if __name__ == "__main__":
    matrix = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])

    starting_position = 8
    final_position = 2

    mdp = MDP_GridSearch(matrix, starting_position, final_position, random_policy=False)
    episodes = 1000
    steps = 20
    best_pol, Q = dqn(mdp, steps=steps, episodes=episodes)
    print(f"Best policy, DQN: {best_pol}")
    print(f"Q found: {Q}")
