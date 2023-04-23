import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import cm
from matplotlib.colors import ListedColormap
import random
import tqdm
from matplotlib import animation
from typing import List, Tuple, Union

# ---------------------- Custom ColorMap ----------------------

colors = [
    cm.get_cmap("Oranges_r", 2)(np.linspace(0, 1, 2)),
    cm.get_cmap("Blues", 3)(np.linspace(0.5, 1, 2)),
]
colors = np.vstack(colors)
CustomColorMap = ListedColormap(colors, name="OrangeBlue")


# ---------------------- Gridworld Class ----------------------


class DetrministicGridworld:
    def __init__(
        self,
        n_cols: int = 6,
        n_rows: int = 6,
        walls: List[Tuple[int, int]] = [],
        traps: List[Tuple[int, int]] = [],
        goal: Union[Tuple[int, int], None] = None,
        cost_per_step: float = 0.1,
        goal_reward: float = 2,
        trap_reward: float = -2,
    ) -> None:
        """_summary_

        Args:
            n_cols (int, optional): Number of Columns. Defaults to 6.
            n_rows (int, optional): Number of Rows. Defaults to 6.
            walls (List[Tuple[int, int]]): Cordinates of Walls. Defaults to [].
            traps (List[Tuple[int, int]]): Cordinates of Traps. They "kill" your agent. Defaults to [].
            goal (Union[Tuple[int, int], None], optional): Cordinates of goal. Defaults to bottem right corner.
            cost_per_step (float, optional): Cost per single step. Defaults to 0.1.
            goal_reward (int, optional): Reward of goal. Defaults to 2.
            trap_reward (int, optional): Reward of trap. Defaults to -2.
        """

        # ----------------- Save Atributes -----------------

        self.dimensions = (n_cols, n_rows)
        self.walls = walls

        # if none default to bottem right corner
        if goal is None:
            self.goal = (self.dimensions[0] - 1, self.dimensions[1] - 1)
        else:
            self.goal = goal

        self.cost_per_step = cost_per_step
        self.goal_reward = goal_reward
        self.traps = traps
        self.trap_costs = trap_reward

        self._build_board()

    def sample_trajectory(self, start_state):
        """Implement a trajectory"""

        reward = 0

        position = start_state

        # while not dead or reached goal

        while position != self.goal and position not in self.traps:
            state_policy = self.policy[position]

            # sample next postion

            next_postion = random.choices(
                population=list(state_policy.keys()),
                weights=list(state_policy.values()),
            )[0]
            position = self.direction_to_cord(next_postion, position)

            # reward for step penalty
            reward -= self.cost_per_step

        # apply final reward

        if position in self.traps:
            reward += self.trap_costs

        else:
            reward += self.goal_reward

        return reward

    def episode(
        self,
    ):
        # draw random starting state

        state = random.choice(self.all_states)

        # get reward of trajectory

        sampled_reward = self.sample_trajectory(state)

        # update values

        aggregator, counter = self.state_values[state]

        self.state_values[state] = (aggregator + sampled_reward, counter + 1)

    def run(self, viz_at=[50, 200, 500, 1000, 10000]):

        # agg for calculated values

        captured_values = {}

        for i in tqdm.tqdm(range(max(viz_at)), desc="Simulate"):
            self.episode()

            # save values
            if i + 1 in viz_at:
                captured_values[i + 1] = self.get_state_values().copy()

        for n, values in captured_values.items():
            self.visualize(f"After {n} Iterations", values)

    def _build_board(self):
        """Builds the internal bord representation"""

        # dict for directions
        directions = ["left", "right", "up", "down"]
        cords = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.directions = dict(zip(directions, cords))

        # set of all reachable tiles
        self.all_valid_tiles = {
            cords for cords in np.ndindex(self.dimensions) if cords not in self.walls
        }

        # set of all states
        self.all_states = list(
            self.all_valid_tiles.difference([self.goal]).difference(self.traps)
        )

        # aggregator for means
        self.state_values = {cord: (0, 0) for cord in self.all_states}

        # function to return all valid neighbor directions
        def get_neighbors(cords):
            return [
                name
                for name in self.directions
                if self.direction_to_cord(name, cords) in self.all_valid_tiles
            ]

        # dict of all valid neighbors
        self.neighbors = {
            cords: get_neighbors(cords) for cords in self.all_states
        }

        # initiate policy

        self.policy = {}

        for cords in self.all_states:
            # get direction
            possible_directions = self.neighbors[cords]

            # argmin of manhattan distance
            best_direction = min(
                possible_directions,
                key=lambda x: self.manhatten_distance_to_goal(
                    self.direction_to_cord(x, cords)),
            )

            # best directions get 80% prob
            state_policy = {best_direction: 0.8}

            # the rest get 20% split up evenly

            for dir in possible_directions:
                if dir == best_direction:
                    continue

                state_policy[dir] = 0.2 / (len(possible_directions) - 1)

            self.policy[cords] = state_policy

    def get_state_values(self):
        """Retunrs a Dictionary with every state and it's Markov Policy Value"""

        states = {}

        for cords, (sum, n) in self.state_values.items():
            states[cords] = sum / n if n > 0 else "NA"

        return states

    def manhatten_distance_to_goal(self, cords):
        """Small util"""
        x, y = cords

        return abs(self.goal[0] - x) + abs(self.goal[1] - y)

    def direction_to_cord(self, direction, cord):
        """Small util"""

        diff = self.directions[direction]

        return cord[0] + diff[0], cord[1] + diff[1]

    def plot_arrows(self):
        for cords, state_policies in self.policy.items():
            x, y = cords

            for direction, prob in state_policies.items():
                width_factor = 1 if prob == 0.8 else 0.37

                buffer = 0.24 if prob == 0.8 else 0.3

                kwargs = {
                    "width": 0.022 * width_factor,
                    "length_includes_head": False,
                    "head_width": 0.1 * width_factor,
                    "head_length": 0.05,
                    "color": "black",
                }

                if direction == "right":
                    plt.arrow(x + buffer, y, 0.17 * width_factor, 0, **kwargs)

                elif direction == "left":
                    plt.arrow(x - buffer, y, -0.17 * width_factor, 0, **kwargs)

                elif direction == "up":
                    plt.arrow(x, y - buffer, 0, -0.17 * width_factor, **kwargs)

                elif direction == "down":
                    plt.arrow(x, y + buffer, 0, 0.17 * width_factor, **kwargs)

    def visualize(self, title="Initial Gridworld", values=None):
        if values is None:
            values = self.get_state_values()

        image = np.ones(self.dimensions[::-1])

        # image[self.start] = 2
        image[self.goal[::-1]] = 3

        walls_x_cords = [x for x, y in self.walls]
        walls_y_cords = [y for x, y in self.walls]

        image[walls_y_cords, walls_x_cords] = 0

        traps_x_cords = [x for x, y in self.traps]
        traps_y_cords = [y for x, y in self.traps]

        image[traps_y_cords, traps_x_cords] = 2

        plt.figure(
            figsize=(self.dimensions[0] * 1.5, self.dimensions[1] * 1.5))

        plt.imshow(image, CustomColorMap)

        for i in range(self.dimensions[0] - 1):
            plt.axvline(i + 0.5, color="black", linewidth=0.7)

        for i in range(self.dimensions[1] - 1):
            plt.axhline(i + 0.48, color="black", linewidth=0.7)

        for (x, y), val in values.items():
            if type(val) == str:
                pass

            else:
                val = f"{val:.2f}"

            plt.text(x, y, val, ha="center", va="center", size=14)

        for x, y in self.traps:
            plt.text(x, y, self.trap_costs, ha="center", va="center", size=14)

        plt.text(
            self.goal[0],
            self.goal[1],
            self.goal_reward,
            ha="center",
            va="center",
            size=14,
        )

        self.plot_arrows()

        plt.title(title)

        legend_elements = [
            Patch(facecolor=CustomColorMap(0), label="Walls"),
            Patch(facecolor=CustomColorMap(1), label="Tiles"),
            Patch(facecolor=CustomColorMap(2), label="Trap"),
            Patch(facecolor=CustomColorMap(3), label="Goal"),
        ]

        plt.legend(handles=legend_elements, bbox_to_anchor=(1.15, 1.01))
        plt.show()

    def animate_episode(
        self, title="Initial Gridworld", values=None, start_point=(0, 0), nbuffer=10
    ):
        if values is None:
            values = self.get_state_values()

        image = np.ones(self.dimensions[::-1])

        # image[self.start] = 2
        image[self.goal] = 3

        walls_x_cords = [x for x, y in self.walls]
        walls_y_cords = [y for x, y in self.walls]

        image[walls_y_cords, walls_x_cords] = 0

        traps_x_cords = [x for x, y in self.traps]
        traps_y_cords = [y for x, y in self.traps]

        image[traps_y_cords, traps_x_cords] = 2

        fig = plt.figure(
            figsize=(self.dimensions[0] * 1.5, self.dimensions[1] * 1.5))

        plt.imshow(image, CustomColorMap)

        for i in range(self.dimensions[0] - 1):
            plt.axvline(i + 0.5, color="black", linewidth=0.7)

        for i in range(self.dimensions[1] - 1):
            plt.axhline(i + 0.48, color="black", linewidth=0.7)

        for (x, y), val in values.items():
            if type(val) == str:
                pass

            else:
                val = f"{val:.2f}"

            plt.text(x, y, val, ha="center", va="center", size=14)

        for x, y in self.traps:
            plt.text(x, y, self.trap_costs, ha="center", va="center", size=14)

        plt.text(
            self.goal[0],
            self.goal[1],
            self.goal_reward,
            ha="center",
            va="center",
            size=14,
        )

        ax = plt.gca()

        self.plot_arrows()
        ttl = plt.text(
            0.5,
            1.05,
            f"Current Reward: {0:.2f}",
            transform=ax.transAxes,
            # ha="center",
            # va="center",
            fontsize=12,
        )

        legend_elements = [
            Patch(facecolor=CustomColorMap(0), label="Walls"),
            Patch(facecolor=CustomColorMap(1), label="Tiles"),
            Patch(facecolor=CustomColorMap(2), label="Trap"),
            Patch(facecolor=CustomColorMap(3), label="Goal"),
        ]
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.15, 1.01))

        state = (0, 0)

        (agent,) = plt.plot(state[0], state[1], "-ro", markersize=25)

        reward = 0

        position = start_point

        states = [position]
        rewards = [0]

        while position != self.goal and position not in self.traps:
            state_policy = self.policy[position]

            next_postion = random.choices(
                population=list(state_policy.keys()),
                weights=list(state_policy.values()),
            )[0]
            position = self.direction_to_cord(next_postion, position)
            reward -= self.cost_per_step
            rewards.append(reward)
            states.append(position)

        if position in self.traps:
            reward += self.trap_costs

        else:
            reward += self.goal_reward

        rewards[-1] = reward

        def update(i):
            complete = i % nbuffer == 0

            rounded = i // nbuffer

            ttl.set_text(
                f"Current Reward: {rewards[rounded]:.2f}",
            )

            if complete:
                cords = np.array(states[rounded])
                agent.set_data(cords)
                return (agent, ttl)

            next_point = np.array(states[rounded + 1])
            last = np.array(states[rounded])
            diff = next_point - last

            cords = last + (i % nbuffer) / (nbuffer - 1) * diff
            agent.set_data(cords)
            return (agent, ttl)

        steps = np.arange(1, (len(states) - 1) * nbuffer)

        ani = animation.FuncAnimation(fig, update, steps)

        return ani
