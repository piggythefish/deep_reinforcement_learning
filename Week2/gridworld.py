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
    cm.get_cmap("Oranges_r")(np.linspace(0, 1, 3)),
    cm.get_cmap("Blues")(np.linspace(0.5, 1, 2)),
]
colors = np.vstack(colors)
CustomColorMap = ListedColormap(colors, name="OrangeBlue")


# ---------------------- Gridworld Class ----------------------


class ProbabilisticGridWorld:
    def __init__(
        self,
        n_cols: int = 6,
        n_rows: int = 6,
        walls: List[Tuple[int, int]] = [],
        traps: List[Tuple[int, int]] = [],
        start: Tuple[int, int] = (0, 0),
        goal: Union[Tuple[int, int], None] = None,
        cost_per_step: float = 0.1,
        goal_reward: float = 2,
        trap_reward: float = -2,
        max_steps_per_episode=200,
        gamma=0.9,
        probality_wrong_step=0.2,
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
        self.start = start
        self.gamma = gamma
        self.max_steps_per_episode = max_steps_per_episode
        self.probality_wrong_step = probality_wrong_step

        self._build_board()

    def create_episode(self, start_state):
        """Implement a trajectory"""

        position = start_state

        # while not dead or reached goal

        history = []

        n_steps = 0

        while position != self.goal and position not in self.traps and n_steps < self.max_steps_per_episode:

            best_direction = self.policy[position]

            imediate_reward, next_position = self.sample_from_environment(
                position, best_direction)

            history.append((position, imediate_reward))

            position = next_position

            n_steps += 1

        return history

    def step(
        self,
    ):
        # draw random starting state

        state = self.start

        # get reward of trajectory

        history = self.create_episode(state)

        # update values

        self.update_values(history)

        avg_return_per_episode = self.get_state_value(self.start)

        return avg_return_per_episode

    def update_values(self, history):

        visisited = set()

        gamma_weigted_return = 0

        while len(history):

            position, imediate_reward = history.pop()

            if position in visisited:

                continue

            # visisited.add(position)

            new_state_value = imediate_reward + self.gamma * gamma_weigted_return

            self.aggregate_state_value(position, new_state_value)

            gamma_weigted_return = imediate_reward + \
                self.gamma * gamma_weigted_return

            neighbors = self.neighbors[position]

            greedy_policy = max(neighbors,
                                key=lambda x: self.q_function(position, x))

            self.policy[position] = greedy_policy

    def aggregate_state_value(self, state, new_val):

        aggregator, counter = self.state_values[state]
        self.state_values[state] = (
            aggregator + new_val, counter + 1)

    def q_function(self, state, action):

        imediate_reward = self.reward_dynamics[(state, action)]
        poss_states, probs = self.state_dynamics[(state, action)]

        return_value = imediate_reward

        for next_state, prob in zip(poss_states, probs):

            if next_state in self.state_values:

                V_next_state = self.get_state_value(next_state)

            else:
                V_next_state = 0

            return_value += self.gamma * V_next_state * prob

        return return_value

    def sample_from_environment(self, state, action):

        reward = self.reward_dynamics[(state, action)]
        poss_states, probs = self.state_dynamics[(state, action)]

        next_position = random.choices(
            population=poss_states, weights=probs)[0]

        return reward, next_position

    def get_state_value(self, state):

        sum, n = self.state_values[state]

        val = sum / n if n > 0 else 0

        return val

    def get_state_values(self):
        """Retunrs a Dictionary with every state and it's Markov Policy Value"""

        states = {}

        for cords in self.state_values.keys():

            states[cords] = self.get_state_value(cords)

        return states

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
        self.state_values = {cord: (0.1, 1) for cord in self.all_states}

        # function to return all valid neighbor directions
        def get_neighbors(cords):
            return [
                name
                for name in self.directions
                if self.direction_to_cord(name, cords) in self.all_valid_tiles
            ]

        # dict of all valid neighbors
        self.neighbors = {cords: get_neighbors(
            cords) for cords in self.all_states}

        # initiate policy

        self.policy = {}
        self.reward_dynamics = {}
        self.state_dynamics = {}

        for cords in self.all_states:
            # get direction
            possible_directions = self.neighbors[cords]

            # random initialization
            direction = random.choice(possible_directions)

            self.policy[cords] = direction

            # create state and reward dynamics
            avg_rewards_for_action = {}

            for action in possible_directions:

                reward_of_action = self.compute_simple_reward(cords, action)

                prob_correct = (1 - self.probality_wrong_step)
                resulting_state = self.direction_to_cord(action, cords)

                state_dynamic = {resulting_state: prob_correct}

                avg = prob_correct * reward_of_action

                other_actions = set(possible_directions).difference((action,))

                prob_false = (self.probality_wrong_step / len(other_actions))

                for other_action in other_actions:
                    reward_of_action = self.compute_simple_reward(
                        cords, other_action)

                    avg += prob_false * reward_of_action

                    resulting_state = self.direction_to_cord(
                        other_action, cords)
                    state_dynamic[resulting_state] = prob_false

                self.state_dynamics[(cords, action)] = list(
                    state_dynamic.keys()), list(state_dynamic.values())
                self.reward_dynamics[(cords, action)] = avg

    def compute_simple_reward(self, state, action):

        reward = -self.cost_per_step

        next_position = self.direction_to_cord(action, state)

        if next_position in self.traps:
            reward += self.trap_costs

        elif next_position == self.goal:
            reward += self.goal_reward

        return reward

    def direction_to_cord(self, direction, cord):
        """Small util"""

        diff = self.directions[direction]

        return cord[0] + diff[0], cord[1] + diff[1]

    def plot_arrows(self):

        for cords, direction in self.policy.items():
            x, y = cords

            buffer = 0.24

            kwargs = {
                "width": 0.022,
                "length_includes_head": False,
                "head_width": 0.1,
                "head_length": 0.05,
                "color": "black",
            }

            if direction == "right":
                plt.arrow(x + buffer, y, 0.17, 0, **kwargs)

            elif direction == "left":
                plt.arrow(x - buffer, y, -0.17, 0, **kwargs)

            elif direction == "up":
                plt.arrow(x, y - buffer, 0, -0.17, **kwargs)

            elif direction == "down":
                plt.arrow(x, y + buffer, 0, 0.17, **kwargs)

    def visualize(self, title="Initial State of GridWorld", values=None, savefig=False):
        if values is None:
            values = self.get_state_values()

        image = np.ones(self.dimensions[::-1]) + 1

        image[self.start[::-1]] = 1

        image[self.goal[::-1]] = 4

        walls_x_cords = [x for x, y in self.walls]
        walls_y_cords = [y for x, y in self.walls]

        image[walls_y_cords, walls_x_cords] = 0

        traps_x_cords = [x for x, y in self.traps]
        traps_y_cords = [y for x, y in self.traps]

        image[traps_y_cords, traps_x_cords] = 3

        plt.figure(
            figsize=(self.dimensions[0] * 1.5, self.dimensions[1] * 1.5))

        plt.imshow(image, CustomColorMap)

        for i in range(self.dimensions[0] - 1):
            plt.axvline(i + 0.49, color="black", linewidth=1)

        for i in range(self.dimensions[1] - 1):
            plt.axhline(i + 0.49, color="black", linewidth=1)

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
        plt.xticks(fontsize=14,)
        plt.yticks(fontsize=14,)

        plt.title(title, fontsize=17)

        legend_elements = [
            Patch(facecolor=CustomColorMap(0), label="Walls"),
            Patch(facecolor=CustomColorMap(1), label="Start"),
            Patch(facecolor=CustomColorMap(2), label="Tiles"),
            Patch(facecolor=CustomColorMap(3), label="Trap"),
            Patch(facecolor=CustomColorMap(4), label="Goal"),
        ]

        plt.legend(handles=legend_elements,
                   bbox_to_anchor=(1.01, 1.01), fontsize=14)

        if savefig:
            plt.savefig(
                "imgs/" + title.lower().replace(" ", "_") + ".jpg", dpi=400, bbox_inches="tight")

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
            size=17,
        )

        ax = plt.gca()

        self.plot_arrows()
        ttl = plt.text(
            0.5,
            1.05,
            f"Current Reward: {0:.2f}",
            transform=ax.transAxes,
            ha="center",
            # va="center",
            fontsize=17,
        )

        legend_elements = [
            Patch(facecolor=CustomColorMap(0), label="Walls"),
            Patch(facecolor=CustomColorMap(1), label="Tiles"),
            Patch(facecolor=CustomColorMap(2), label="Trap"),
            Patch(facecolor=CustomColorMap(3), label="Goal"),
        ]
        plt.legend(handles=legend_elements,
                   bbox_to_anchor=(0.935, 1.16), fontsize=14)
        plt.xticks(fontsize=14,)
        plt.yticks(fontsize=14,)

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
