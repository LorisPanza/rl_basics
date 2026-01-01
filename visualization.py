import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
import matplotlib.patches as patches
from environment import MDP_GridSearch
from optimal_policy_search import (
    policy_iteration,
    value_iteration,
    montecarlo_model_free_on_policy,
    td_model_free_on_policy,
    q_learning_model_free_off_policy,
)
from value_function_approximator import dqn


class RLVisualizer:
    def __init__(self, mdp, algorithm_name="policy_iteration"):
        """
        Initialize the RL algorithm visualizer.
        :param mdp: MDP environment
        :param algorithm_name: Name of the algorithm to visualize
        """
        self.mdp = mdp
        self.algorithm_name = algorithm_name
        self.fig = None
        self.axes = None
        self.policy_history = []
        self.value_history = []
        self.q_history = []
        self.episode_rewards = []
        self.current_step = 0
        self.colorbars = {}  # Store colorbars for cleanup

    def visualize_grid_policy(self, policy, ax, title="Greedy Policy"):
        """
        Visualize the policy on the grid.
        :param policy: Policy array (deterministic)
        :param ax: Matplotlib axis
        :param title: Title of the plot
        """
        ax.clear()
        grid_size = self.mdp.n

        # Create grid background
        for i in range(grid_size):
            for j in range(grid_size):
                state_idx = i * grid_size + j
                state_num = state_idx + 1
                color = (
                    "lightgreen"
                    if state_num == self.mdp.final_position
                    else "lightblue"
                )
                if state_num == self.mdp.starting_position:
                    color = "lightyellow"

                rect = patches.Rectangle(
                    (j, grid_size - i - 1),
                    1,
                    1,
                    linewidth=1,
                    edgecolor="black",
                    facecolor=color,
                )
                ax.add_patch(rect)

        # Draw policy arrows
        if isinstance(policy, np.ndarray) and len(policy) > 0:
            for i in range(grid_size):
                for j in range(grid_size):
                    state_idx = i * grid_size + j
                    if state_idx < len(policy):
                        action = policy[state_idx]
                        action_symbols = {0: "↑", 1: "↓", 2: "←", 3: "→"}
                        ax.text(
                            j + 0.5,
                            grid_size - i - 1 + 0.5,
                            action_symbols.get(action, "?"),
                            ha="center",
                            va="center",
                            fontsize=16,
                            fontweight="bold",
                        )

        ax.set_xlim(0, grid_size)
        ax.set_ylim(0, grid_size)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")

    def visualize_value_function(self, values, ax, title="Value Function"):
        """
        Visualize the value function as a heatmap.
        :param values: Value array
        :param ax: Matplotlib axis
        :param title: Title of the plot
        """
        # Remove previous colorbar if it exists
        if "value" in self.colorbars and self.colorbars["value"] is not None:
            self.colorbars["value"].remove()

        ax.clear()
        grid_size = self.mdp.n
        value_grid = values.reshape((grid_size, grid_size))

        im = ax.imshow(value_grid, cmap="YlOrRd", interpolation="nearest")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(range(grid_size))
        ax.set_yticks(range(grid_size))

        # Add value text on cells
        for i in range(grid_size):
            for j in range(grid_size):
                ax.text(
                    j,
                    i,
                    f"{value_grid[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=10,
                )

        self.colorbars["value"] = plt.colorbar(im, ax=ax, label="Value")

    def visualize_training_progress(self, ax, rewards, title="Training Progress"):
        """
        Visualize training progress over episodes.
        :param ax: Matplotlib axis
        :param rewards: List of episode rewards
        :param title: Title of the plot
        """
        ax.clear()
        if len(rewards) > 0:
            ax.plot(rewards, linewidth=2, color="blue", marker="o", markersize=4)
            ax.fill_between(range(len(rewards)), rewards, alpha=0.3)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Cumulative Reward")
        ax.grid(True, alpha=0.3)

    def create_dashboard(self, steps=20, num_episodes=100):
        """
        Create a comprehensive dashboard for RL algorithm visualization.
        :param steps: Number of steps per episode
        :param num_episodes: Number of episodes to train
        """
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle(
            f"RL Algorithm Visualization: {self.algorithm_name}",
            fontsize=16,
            fontweight="bold",
        )

        # Create subplots
        gs = self.fig.add_gridspec(
            2, 3, hspace=0.3, wspace=0.4, width_ratios=[1, 1, 0.35]
        )
        ax_policy = self.fig.add_subplot(gs[0, 0])
        ax_value = self.fig.add_subplot(gs[0, 1])
        ax_q_heatmap = self.fig.add_subplot(gs[1, :-1])
        ax_info = self.fig.add_subplot(gs[:, 2])

        self.axes = {
            "policy": ax_policy,
            "value": ax_value,
            "q_heatmap": ax_q_heatmap,
            "info": ax_info,
        }

        # Run algorithm and collect history
        print(f"Training {self.algorithm_name}...")
        self._run_algorithm_with_history(steps, num_episodes)

        # Initial visualization
        self._update_visualizations(0)

        # Create slider for step-through
        ax_slider = self.fig.add_axes([0.2, 0.02, 0.6, 0.03])
        slider = Slider(
            ax_slider, "Step", 0, len(self.policy_history) - 1, valinit=0, valstep=1
        )

        def update_step(val):
            step = int(slider.val)
            self._update_visualizations(step)

        slider.on_changed(update_step)

        plt.show()

    def _run_algorithm_with_history(self, steps, num_episodes):
        """
        Run the selected algorithm and collect history for visualization.
        """
        if self.algorithm_name == "policy_iteration":
            policy, values, pol_hist, val_hist = policy_iteration(self.mdp)
            self.policy_history = pol_hist
            self.value_history = val_hist

        elif self.algorithm_name == "value_iteration":
            policy, values, pol_hist, val_hist = value_iteration(self.mdp)
            self.policy_history = pol_hist
            self.value_history = val_hist

        elif self.algorithm_name == "montecarlo":
            policy, Q, pol_hist, Q_hist = montecarlo_model_free_on_policy(
                self.mdp, steps=steps, num_episodes=num_episodes
            )
            self.policy_history = pol_hist
            self.q_history = Q_hist
            self.value_history = [np.max(Q_val, axis=1) for Q_val in Q_hist]

        elif self.algorithm_name == "td_sarsa":
            policy, Q, pol_hist, Q_hist = td_model_free_on_policy(
                self.mdp, steps=steps, num_episodes=num_episodes
            )
            self.policy_history = pol_hist
            self.q_history = Q_hist
            self.value_history = [np.max(Q_val, axis=1) for Q_val in Q_hist]

        elif self.algorithm_name == "q_learning":
            policy, Q, pol_hist, Q_hist = q_learning_model_free_off_policy(
                self.mdp, steps=steps, num_episodes=num_episodes
            )
            self.policy_history = pol_hist
            self.q_history = Q_hist
            self.value_history = [np.max(Q_val, axis=1) for Q_val in Q_hist]
        """
            elif self.algorithm_name == "dqn":
            Q, policy = dqn(self.mdp, steps=steps, episodes=num_episodes)
            self.policy_history = [policy]
            self.q_history = [Q]
            self.value_history = [np.max(Q, axis=1)]
        """

    def _update_visualizations(self, step):
        """
        Update all visualizations for a given step.
        """
        step = min(step, len(self.policy_history) - 1)

        # Policy visualization
        if self.policy_history:
            self.visualize_grid_policy(
                self.policy_history[step], self.axes["policy"], f"Policy (Step {step})"
            )

        # Value function visualization
        if self.value_history:
            self.visualize_value_function(
                self.value_history[step],
                self.axes["value"],
                f"Value Function (Step {step})",
            )

        # Q-value heatmap
        if self.q_history and len(self.q_history) > step:
            # Remove previous colorbar if it exists
            if (
                "q_heatmap" in self.colorbars
                and self.colorbars["q_heatmap"] is not None
            ):
                self.colorbars["q_heatmap"].remove()

            ax = self.axes["q_heatmap"]
            ax.clear()
            Q = self.q_history[step]
            im = ax.imshow(Q, cmap="viridis", aspect="auto")
            ax.set_title(
                f"Q-Values Heatmap (Step {step})", fontsize=12, fontweight="bold"
            )
            ax.set_xlabel("Action")
            ax.set_ylabel("State")
            ax.set_xticks(range(len(self.mdp.action_space)))
            ax.set_xticklabels(["↑", "↓", "←", "→"])
            self.colorbars["q_heatmap"] = plt.colorbar(im, ax=ax, label="Q-Value")
        else:
            # Hide the Q-heatmap axis if no Q history is available
            ax = self.axes["q_heatmap"]
            ax.clear()
            ax.axis("off")

        # Info panel
        ax_info = self.axes["info"]
        ax_info.clear()
        ax_info.axis("off")
        info_text = f"""
        Algorithm: {self.algorithm_name}
        Step: {step}
        State Space: {len(self.mdp.state_space)}
        Action Space: {len(self.mdp.action_space)}
        Grid Size: {self.mdp.n}×{self.mdp.n}
        Starting Position: {self.mdp.starting_position}
        Final Position: {self.mdp.final_position}
        """
        ax_info.text(
            0.1,
            0.5,
            info_text,
            fontsize=11,
            verticalalignment="center",
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        self.fig.canvas.draw_idle()


def visualize_algorithm_comparison(mdp, steps=20, num_episodes=100):
    """
    Compare multiple RL algorithms side by side.
    :param mdp: MDP environment
    :param steps: Number of steps per episode
    :param num_episodes: Number of episodes to train
    """
    algorithms = ["policy_iteration", "value_iteration", "q_learning"]

    fig, axes = plt.subplots(len(algorithms), 2, figsize=(14, 12))
    fig.suptitle("RL Algorithms Comparison", fontsize=16, fontweight="bold")

    for idx, algo in enumerate(algorithms):
        print(f"Running {algo}...")
        visualizer = RLVisualizer(mdp, algo)
        visualizer._run_algorithm_with_history(steps, num_episodes)

        if visualizer.policy_history:
            visualizer.visualize_grid_policy(
                visualizer.policy_history[0],
                axes[idx, 0],
                f"{algo.replace('_', ' ').title()} - Policy",
            )

        if visualizer.value_history:
            visualizer.visualize_value_function(
                visualizer.value_history[0],
                axes[idx, 1],
                f"{algo.replace('_', ' ').title()} - Value",
            )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    matrix = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]])

    starting_position = 3
    final_position = 7

    mdp = MDP_GridSearch(matrix, starting_position, final_position, random_policy=True)

    # Single algorithm visualization with step-through slider
    visualizer = RLVisualizer(mdp, algorithm_name="q_learning")
    visualizer.create_dashboard(steps=30, num_episodes=1000)

    # Uncomment to compare algorithms
    # visualize_algorithm_comparison(mdp, steps=20, num_episodes=100)
