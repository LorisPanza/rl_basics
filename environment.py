import numpy as np


class MDP_GridSearch:
    def __init__(
        self,
        matrix: np.array,
        starting_position: int,
        final_position: int,
        random_policy: bool = True,
    ):
        """
        Initialize the MDP environment with a grid matrix, starting position, final position, and policy
        :param matrix: 2D numpy array representing the grid and the reward associated with each state
        :param starting_position: Starting state (1-based index)
        :param final_position: Final state (1-based index)
        :param random_policy: If True, a random policy will be generated; otherwise, a deterministic policy will be used
        """
        self.matrix = matrix  # Save for later use # N x M
        self.random_policy = random_policy
        self.n = matrix.shape[0]
        self.state_space = np.arange(
            1, self.n * self.n + 1
        )  # (N*M) = S TODO: make 0-based?
        self.action_space = ["up", "down", "left", "right"]  # (A)
        self.map_action = {"up": 0, "down": 1, "left": 2, "right": 3}  # (A)
        self.map_number_to_action = {0: "up", 1: "down", 2: "left", 3: "right"}
        self.rew = self.compute_rewards()

        if self.random_policy:
            self.policy = np.random.rand(len(self.state_space), len(self.action_space))
            self.policy = self.policy / self.policy.sum(
                axis=1, keepdims=True
            )  # Normalize policy
            """
            [[0.22465905 0.5795156  0.07653217 0.11929318]
            [0.3705597  0.32106803 0.26160365 0.04676862]
            [0.13525356 0.0404906  0.06064916 0.76360668]
            [0.13809877 0.1971799  0.30556573 0.3591556 ]
            [0.24916384 0.21498783 0.30888283 0.2269655 ]
            [0.30294602 0.48790565 0.16959502 0.0395533 ]
            [0.2900728  0.0276925  0.34935798 0.33287672]
            [0.3105762  0.23794996 0.16804188 0.28343196]
            [0.24864597 0.04569897 0.21153989 0.49411517]]
            """
            self.policy = np.array(
                [
                    [0.22465905, 0.5795156, 0.07653217, 0.11929318],
                    [0.3705597, 0.32106803, 0.26160365, 0.04676862],
                    [0.13525356, 0.0404906, 0.06064916, 0.76360668],
                    [0.13809877, 0.1971799, 0.30556573, 0.3591556],
                    [0.24916384, 0.21498783, 0.30888283, 0.2269655],
                    [0.30294602, 0.48790565, 0.16959502, 0.0395533],
                    [0.2900728, 0.0276925, 0.34935798, 0.33287672],
                    [0.3105762, 0.23794996, 0.16804188, 0.28343196],
                    [0.24864597, 0.04569897, 0.21153989, 0.49411517],
                ]
            )
        else:
            self.policy = [0, 0, 0, 0, 0, 0, 0, 0, 0]

        print("Reward map: ")
        print(self.matrix)  # N x M

        print("Policy:")
        print(self.policy)  # if not random -> S, else S x A

        print("Reward matrix: ")
        print(self.rew)  # S x A

        print("State space: ")
        print(self.state_space)  # S

        # self.rew_pi = self.state_reward_policy()
        # print("Reward for each state given a policy: ")
        # print(self.rew_pi) # S

        assert starting_position in self.state_space
        assert final_position in self.state_space

        self.starting_position = starting_position
        self.final_position = final_position

        print(f"Init pos {self.starting_position}, Final pos: {self.final_position}")

        self.actual_state = self.starting_position

        # Parametric movement matrices
        size = self.n * self.n
        self.right = np.zeros((size, size), dtype=int)
        self.left = np.zeros((size, size), dtype=int)
        self.up = np.zeros((size, size), dtype=int)
        self.down = np.zeros((size, size), dtype=int)

        for i in range(size):
            row, col = divmod(i, self.n)
            # Right
            if col < self.n - 1:
                self.right[i, i + 1] = 1
            # Left
            if col > 0:
                self.left[i, i - 1] = 1
            # Up
            if row > 0:
                self.up[i, i - self.n] = 1
            # Down
            if row < self.n - 1:
                self.down[i, i + self.n] = 1

        self.action_matrix = {0: self.up, 1: self.down, 2: self.left, 3: self.right}

    def compute_rewards(self):
        # Create a reward matrix based on the provided matrix
        rew_matrix = np.zeros((len(self.state_space), len(self.action_space)))
        for i in range(len(self.state_space)):
            for j in range(len(self.action_space)):
                rew_matrix[i, j] = self.rewards(
                    self.state_space[i], self.action_space[j]
                )
        return rew_matrix

    def rewards(self, state, action):
        # Convert state (1-based) to (row, col)
        idx = state - 1
        row, col = divmod(idx, self.n)
        # Compute next state based on action
        if action == "up" and row > 0:
            row -= 1
        elif action == "down" and row < self.n - 1:
            row += 1
        elif action == "left" and col > 0:
            col -= 1
        elif action == "right" and col < self.n - 1:
            col += 1
        # Return reward from matrix
        return self.matrix[row, col]

    def state_reward_policy(self):
        # Calculate the expected reward for each state given the policy
        rew_pi = np.zeros(len(self.state_space))
        if self.random_policy:
            for s in range(len(self.state_space)):
                for a in range(len(self.action_space)):
                    rew_pi[s] += self.policy[s, a] * self.rew[s, a]
        else:
            for s in range(len(self.state_space)):
                a = self.policy[s]
                rew_pi[s] = self.rew[s, a]

        return rew_pi

    def generate_episode(self, steps):
        current_pos_i = self.starting_position
        i = 0
        episode = []

        while current_pos_i != self.final_position and i < steps:
            action_i = np.random.choice(
                self.action_space, p=self.policy[current_pos_i - 1]
            )
            # print(f"Current position: {current_pos_i}; probabilities -> {self.policy[current_pos_i-1]}")

            action_i_num = self.map_action[action_i]
            rew_i = self.rew[current_pos_i - 1, action_i_num]
            i += 1

            next_state_prob_sum = self.action_matrix[action_i_num][
                current_pos_i - 1
            ].sum()

            if next_state_prob_sum == 0:
                next_state_i = current_pos_i
            else:
                next_state_i = (
                    self.action_matrix[action_i_num][current_pos_i - 1].argmax() + 1
                )

            episode.append((current_pos_i, action_i_num, rew_i, next_state_i))
            # print(f"Action: {action_i}, Reward: {rew_i}, Next state: {next_state_i}")
            current_pos_i = next_state_i

        return episode
