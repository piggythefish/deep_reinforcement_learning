import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import cm
from matplotlib.colors import ListedColormap
import random
import tqdm
from matplotlib import animation

colors = [
    cm.get_cmap("Oranges_r", 2)(np.linspace(0, 1, 2)),
    cm.get_cmap("Blues", 3)(np.linspace(0.5, 1, 2)),
]
colors = np.vstack(colors)
newcmp = ListedColormap(colors, name="OrangeBlue")


# firslty deterministic
class Gridworld:
    def __init__(
        self,
        n_cols=6,
        n_rows=6,
        walls=[],
        traps=[],
        start=(0, 0),
        goal=None,
        cost_per_step=0.1,
        goal_reward=2,
        trap_reward=-2,
    ) -> None:
        # ----------------- Save Atributes -----------------

        self.dimensions = (n_cols, n_rows)
        self.walls = walls
        self.start = start

        if goal is None:
            self.goal = (self.dimensions[0] - 1, self.dimensions[1] - 1)
        else:
            self.goal = goal

        directions = ["left", "right", "up", "down"]
        cords = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        self.directions = dict(zip(directions, cords))

        self.cost_per_step = cost_per_step

        self.agent_pos = start
        self.agent_reward = 0
        self.goal_reward = goal_reward
        self.traps = traps
        self.trap_costs = trap_reward

        self.build_board()

    def distance_to_goal(self, cords):
        x, y = cords

        return abs(self.goal[0] - x) + abs(self.goal[1] - y)

    def direction_to_cord(self, direction, cord):
        diff = self.directions[direction]

        return cord[0] + diff[0], cord[1] + diff[1]

    def sample_trajectory(self, start_state):
        reward = 0

        position = start_state

        while position != self.goal and position not in self.traps:
            state_policy = self.policy[position]

            next_postion = random.choices(
                population=list(state_policy.keys()),
                weights=list(state_policy.values()),
            )[0]
            position = self.direction_to_cord(next_postion, position)
            reward -= self.cost_per_step

        if position in self.traps:
            reward += self.trap_costs

        else:
            reward += self.goal_reward

        return reward

    def step(
        self,
    ):
        # draw random starting state

        state = random.choice(self.all_normal_tiles)

        sampled_reward = self.sample_trajectory(state)

        # update values

        aggregator, counter = self.state_values[state]

        self.state_values[state] = (aggregator + sampled_reward, counter + 1)

    def run(self, viz_at=[50, 200, 500, 1000, 10000]):
        captured_values = {}

        for i in tqdm.tqdm(range(max(viz_at)), desc="Simulate"):
            self.step()

            if i + 1 in viz_at:
                captured_values[i + 1] = self.get_state_values().copy()

        for n, values in captured_values.items():
            self.visualize(f"After {n} Iterations", values)

    def build_board(self):
        # ----------------- Build Board -----------------

        self.all_valid_tiles = {
            cords for cords in np.ndindex(self.dimensions) if cords not in self.walls
        }

        self.all_normal_tiles = list(
            self.all_valid_tiles.difference([self.goal]).difference(self.traps)
        )

        self.state_values = {cord: (0, 0) for cord in self.all_normal_tiles}

        def get_neighbors(cords):
            return [
                name
                for name in self.directions
                if self.direction_to_cord(name, cords) in self.all_valid_tiles
            ]

        self.neighbors = {
            cords: get_neighbors(cords) for cords in self.all_normal_tiles
        }

        # initiate policy

        self.policy = {}

        for cords in self.all_normal_tiles:
            possible_directions = self.neighbors[cords]

            best_direction = min(
                possible_directions,
                key=lambda x: self.distance_to_goal(self.direction_to_cord(x, cords)),
            )

            state_policy = {best_direction: 0.8}

            # distribute the rest

            for dir in possible_directions:
                if dir == best_direction:
                    continue

                state_policy[dir] = 0.2 / (len(possible_directions) - 1)

            self.policy[cords] = state_policy

    def get_state_values(self):
        states = {}

        for cords, (sum, n) in self.state_values.items():
            states[cords] = sum / n if n > 0 else "NA"

        return states

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

        plt.figure(figsize=(self.dimensions[0] * 1.5, self.dimensions[1] * 1.5))

        plt.imshow(image, newcmp)

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
            Patch(facecolor=newcmp(0), label="Walls"),
            Patch(facecolor=newcmp(1), label="Tiles"),
            Patch(facecolor=newcmp(2), label="Trap"),
            Patch(facecolor=newcmp(3), label="Goal"),
        ]

        plt.legend(handles=legend_elements, bbox_to_anchor=(1.25, 1))
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

        fig = plt.figure(figsize=(self.dimensions[0] * 1.5, self.dimensions[1] * 1.5))

        plt.imshow(image, newcmp)

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
            Patch(facecolor=newcmp(0), label="Walls"),
            Patch(facecolor=newcmp(1), label="Tiles"),
            Patch(facecolor=newcmp(2), label="Trap"),
            Patch(facecolor=newcmp(3), label="Goal"),
        ]
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.25, 1))

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
