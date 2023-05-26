import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import cm
from matplotlib.colors import ListedColormap
import random
import tqdm
from matplotlib import animation
from typing import List, Tuple, Union
from matplotlib.colors import LogNorm, Normalize
import seaborn as sns
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

# ---------------------- Custom ColorMaps ----------------------

colors = [
    cm.get_cmap("Oranges_r")(np.linspace(0, 1, 3)),
    cm.get_cmap("Blues")(np.linspace(0.5, 1, 2)),
]
colors = np.vstack(colors)
CustomColorMap = ListedColormap(colors, name="OrangeBlue")


top = cm.get_cmap('Oranges_r', 128)
bottom = cm.get_cmap('Blues', 128)

newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                       bottom(np.linspace(0, 1, 128))))
OranbeBlue = ListedColormap(newcolors[::-1], name='OrangeBlue')

# ---------------------- Gridworld Class ----------------------


class MeanAgregator:

    def __init__(self, init_value=None):

        self.counter = 0
        self.value = 0

        if init_value is not None:

            self += init_value

    def __add__(self, other):

        self.value += other
        self.counter += 1
        return self

    def __call__(self,):

        return self.value / self.counter if self.counter > 0 else "NA"


class GridWorldBase:
    def __init__(
        self,
        n_cols: int = 6,
        n_rows: int = 6,
        walls: List[Tuple[int, int]] = [(2, 2), (2, 3)],
        traps: List[Tuple[int, int]] = [(5, 2)],
        start: Tuple[int, int] = (0, 0),
        goal: Union[Tuple[int, int], None] = None,
        cost_per_step: float = 0.05,
        goal_reward: float = 3,
        trap_reward: float = -1,
        probality_wrong_step: float = 0
    ):
        """
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
        self.probality_wrong_step = probality_wrong_step

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

        # initiate policy and enivronemtn dyna,os

        self.policy = {}
        self.state_dynamics = {}

        for cords in self.all_states:
            # get possible directions
            possible_directions = self.neighbors[cords]

            # random initialization
            direction = random.choice(possible_directions)

            self.policy[cords] = direction

            # for every possible action
            for action in possible_directions:

                # initaate state and reward dynamics

                prob_correct = (1 - self.probality_wrong_step)
                resulting_state = self.direction_to_cord(action, cords)

                # add next state and possiblity to reach it
                state_dynamic = {resulting_state: prob_correct}

                # get the other possible actions
                other_actions = set(possible_directions).difference((action,))

                # their probabilities
                prob_false = (self.probality_wrong_step / len(other_actions))

                for other_action in other_actions:

                    resulting_state = self.direction_to_cord(
                        other_action, cords)
                    # save probs of reaching them
                    state_dynamic[resulting_state] = prob_false

                # save env dynamics
                self.state_dynamics[(cords, action)] = list(
                    state_dynamic.keys()), list(state_dynamic.values())

    def sample_from_environment(self, state_action_pair):
        """Sample the environment gives state and ation pair
        Returns new reward and next state"""

        poss_states, probs = self.state_dynamics[state_action_pair]

        next_position = random.choices(
            population=poss_states, weights=probs)[0]

        reward = self.compute_simple_reward(
            state_action_pair[0], next_state=next_position)

        return reward, next_position

    def compute_simple_reward(self, state, action=None, next_state=None):
        """Small util"""

        reward = -self.cost_per_step

        if next_state is None:

            next_state = self.direction_to_cord(action, state)

        elif action is None:

            pass

        else:

            raise ValueError()

        if next_state in self.traps:
            reward += self.trap_costs

        elif next_state == self.goal:
            reward += self.goal_reward

        return reward

    def direction_to_cord(self, direction, cord):
        """Small util"""

        diff = self.directions[direction]

        return cord[0] + diff[0], cord[1] + diff[1]

    def create_episode(self, start_state):

        raise NotImplementedError()

    def step(
        self,
    ):

        raise NotImplementedError()

    def get_state_values(self):

        raise NotImplementedError()

    def base_viz(self, values,  legend_small=False):
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

        fig = plt.figure(
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

        plt.xticks(fontsize=14,)
        plt.yticks(fontsize=14,)

        legend_elements = [
            Patch(facecolor=CustomColorMap(0), label="Walls"),
            Patch(facecolor=CustomColorMap(1), label="Start"),
            Patch(facecolor=CustomColorMap(2), label="Tiles"),
            Patch(facecolor=CustomColorMap(3), label="Trap"),
            Patch(facecolor=CustomColorMap(4), label="Goal"),
        ]
        if not legend_small:
            plt.legend(handles=legend_elements,
                       bbox_to_anchor=(1.01, 1.01), fontsize=14)

        else:
            plt.legend(handles=legend_elements,
                       bbox_to_anchor=(0.935, 1.16), fontsize=14)

        return fig, image,

    def visualize(self, title="Initial State of GridWorld", values=None, savefig=False):

        self.base_viz(values)
        self.plot_arrows()

        plt.title(title, fontsize=17)
        if savefig:
            plt.savefig(savefig, dpi=400, bbox_inches="tight")

        plt.show()

    def plot_arrows(self):
        raise NotImplementedError()


class SarsaGridWorld(GridWorldBase):
    def __init__(
        self,
        grid_world_kwargs={},
        max_steps_per_episode=200,
        gamma=0.9,
        epsilon=0,
        epsilon_anealing_factor=1,
        alpha=0.2,
        n_steps_sarsa=1,

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

        super(SarsaGridWorld, self).__init__(**grid_world_kwargs)

        # ----------------- Save Atributes -----------------

        self.gamma = gamma
        self.max_steps_per_episode = max_steps_per_episode
        self.epsilon = epsilon
        self.epsilon_anealing_factor = epsilon_anealing_factor
        self.alpha = alpha
        self.n_steps_sarsa = n_steps_sarsa

        self._build_board()

        self.q_values = {}

        for cords in self.all_states:
            # get possible directions
            possible_directions = self.neighbors[cords]

            for action in possible_directions:

                self.q_values[(cords, action)] = 0.1

        self.average_return_per_episode = MeanAgregator()

    def create_episode(self, start_state):
        """Implement a episote"""

        position = start_state

        # while not dead or reached goal

        imediate_rewards = []
        actions_taken = []
        from_states = []

        n_steps = 0

        # while goal/traps not reached and under max iterations

        while position != self.goal and position not in self.traps and n_steps < self.max_steps_per_episode:

            # get policy

            if np.random.uniform(0, 1) > self.epsilon:

                action = self.policy[position]

            else:

                action = random.choice(self.neighbors[position])

            state_action_pair = (position, action)
            # sample environment

            imediate_reward, next_position = self.sample_from_environment(
                state_action_pair)

            imediate_rewards.append(imediate_reward)
            actions_taken.append(action)
            from_states.append(position)

            position = next_position

            n_steps += 1
            self.epsilon *= self.epsilon_anealing_factor

        return imediate_rewards, actions_taken, from_states

    def step(
        self,
    ):
        """One step of MK policy iteration"""

        state = self.start

        # samlpe episode

        history = self.create_episode(state)

        # update values

        episode_reward = self.update_values(history)

        self.average_return_per_episode += episode_reward

        return episode_reward

    def update_values(self, history):

        imediate_rewards, actions_taken, from_states = history

        history_length = len(imediate_rewards)

        for i in range(history_length):

            index = i + self.n_steps_sarsa

            rewards = imediate_rewards[i: index]
            gammas = self.gamma ** np.arange(len(rewards))

            reward = np.dot(rewards, gammas)

            if index < history_length:

                n_state_action_pair = from_states[index], actions_taken[index]
                q_val = self.q_values[n_state_action_pair]

                reward += (self.gamma ** self.n_steps_sarsa) * q_val

            current_state_action_pair = from_states[i], actions_taken[i]

            reward -= self.q_values[current_state_action_pair]

            self.q_values[current_state_action_pair] += self.alpha * reward

            position = from_states[i]

            # update policy by argmax of q function

            actions = self.neighbors[position]

            greedy_policy = max(actions,
                                key=lambda x: self.q_values[(position, x)])

            self.policy[position] = greedy_policy

        return_per_episode = np.dot(
            imediate_rewards, self.gamma ** np.arange(history_length))

        return return_per_episode

    def get_state_values(self):
        """Retunrs a Dictionary with every state and it's Markov Policy Value"""

        state_values = {}

        for cords in self.all_states:
            # get possible directions
            possible_directions = self.neighbors[cords]

            state_values[cords] = max(self.q_values[(cords, action)]
                                      for action in possible_directions)

        return state_values

    def plot_arrows(self, with_max_val=True):

        for cords in self.all_states:
            # get possible directions
            possible_directions = self.neighbors[cords]

            vals = np.array([self.q_values[(cords, action)]
                            for action in possible_directions])

            vals -= vals.min()

            total = vals.max()
            argmax = np.argmax(vals)

            if total == 0:
                total += 0.5

            x, y = cords

            for direction, val, i in zip(possible_directions, vals, range(len(vals))):

                part = val / total

                buffer = 0.24

                color = "black" if with_max_val == False or i != argmax else "red"

                kwargs = {
                    "width": 0.022,
                    "length_includes_head": False,
                    "head_width": 0.1,
                    "head_length": 0.05,
                    "color": color,
                }

                length = (part * 0.13) + 0.04

                if direction == "right":
                    plt.arrow(x + buffer, y, length, 0, **kwargs)

                elif direction == "left":
                    plt.arrow(x - buffer, y, -length, 0, **kwargs)

                elif direction == "up":
                    plt.arrow(x, y - buffer, 0, -length, **kwargs)

                elif direction == "down":
                    plt.arrow(x, y + buffer, 0, length, **kwargs)

    def animate_values(self, FREQ=20, N=7000):

        vals = [self.get_state_values().copy()]
        q_vals = [self.q_values.copy()]
        iterations = [0]

        for i in tqdm.tqdm(range(N), desc="Doing Policy Iterations"):
            self.step()

            if i % FREQ == 0:

                vals.append(self.get_state_values().copy())
                q_vals.append(self.q_values.copy())
                iterations.append(i)

        vmin = min(map(lambda x: min(x.values()), vals))
        vmax = max(map(lambda x: max(x.values()), vals))

        image = np.zeros(self.dimensions[::-1]) + vmin

        frames = []

        for state_values, step in zip(vals, iterations):

            for (x, y), v in state_values.items():

                image[y, x] = v

            frames.append((image.copy(), step))

        image = frames[0][0]

        fig = plt.figure(
            figsize=(self.dimensions[0] * 1.5, self.dimensions[1] * 1.2))

        ax = plt.gca()

        ax1_divider = make_axes_locatable(ax)
        im = plt.imshow(image, OranbeBlue, vmin=vmin, vmax=vmax)

        ttl = plt.text(
            0.5,
            1.05,
            f"State Values for Step {0}",
            transform=ax.transAxes,
            ha="center",
            # va="center",
            fontsize=17,
        )

        for y, x in np.ndindex(image.shape):

            text = None

            if (x, y) == self.start:
                text = "Start"

            if (x, y) in self.walls:
                text = "Wall"

            if (x, y) == self.goal:
                text = "Goal"

            if (x, y) in self.traps:
                text = "Trap"

            if text is not None:
                fontcolor = 'black' if image[y, x] > vmin else "white"

                plt.text(x, y, text, ha="center", va="center",
                         size=14, color=fontcolor)

        cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")
        cb1 = plt.colorbar(im, cax=cax1)
        cax1.set_title('State Values', fontsize=12)

        def update(tup):

            frame, i = tup

            im.set_array(frame)

            ttl.set_text(
                f"State Values for Step {i:4d}"
            )

            return im, ttl

        ani = animation.FuncAnimation(fig, update, frames)

        return ani


class MarkovGridWorld(GridWorldBase):
    def __init__(
        self,
        grid_world_kwargs={},
        max_steps_per_episode=200,
        gamma=0.9,
        epsilon=0,
        epsilon_anealing_factor=1

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

        super(MarkovGridWorld, self).__init__(**grid_world_kwargs)

        # ----------------- Save Atributes -----------------

        self.gamma = gamma
        self.max_steps_per_episode = max_steps_per_episode
        self.epsilon = epsilon
        self.epsilon_anealing_factor = epsilon_anealing_factor

        self._build_board()

        # aggregator for means
        self.state_values = {cord: MeanAgregator(
            0.1) for cord in self.all_states}
        self.q_values = {}

        for cords in self.all_states:
            # get possible directions
            possible_directions = self.neighbors[cords]

            for action in possible_directions:

                self.q_values[(cords, action)] = MeanAgregator(0.1)

        self.average_return_per_episode = MeanAgregator()

    def create_episode(self, start_state):
        """Implement a episote"""

        position = start_state

        # while not dead or reached goal

        history = []

        visited_state_action_pairs = set()

        n_steps = 0

        # while goal/traps not reached and under max iterations

        while position != self.goal and position not in self.traps and n_steps < self.max_steps_per_episode:

            # get policy

            if np.random.uniform(0, 1) > self.epsilon:

                action = self.policy[position]

            else:

                action = random.choice(self.neighbors[position])

            state_action_pair = (position, action)
            # sample environment

            imediate_reward, next_position = self.sample_from_environment(
                state_action_pair)

            first_visit = state_action_pair not in visited_state_action_pairs

            # append state and imediate reward

            history.append((position, imediate_reward, action, first_visit))

            visited_state_action_pairs.add(state_action_pair)

            position = next_position

            n_steps += 1
            self.epsilon *= self.epsilon_anealing_factor

        return history

    def step(
        self,
    ):
        """One step of MK policy iteration"""

        state = self.start

        # samlpe episode

        history = self.create_episode(state)

        # update values

        self.update_values(history)

        # get average return per episode

        return_per_episode = self.state_values[self.start]()

        self.average_return_per_episode += return_per_episode

        return return_per_episode

    def update_values(self, history):

        # gamma sum

        gamma_weigted_return = 0

        while len(history):

            # go from back to front
            position, imediate_reward, action, first_visit = history.pop()

            # new state value

            gamma_weigted_return = imediate_reward + self.gamma * gamma_weigted_return

            if first_visit == False:

                continue

            # add to other samples

            self.state_values[position] += gamma_weigted_return

            # update gamma sums

            self.q_values[(position, action)] += gamma_weigted_return

            # update policy by argmax of q function

            neighbors = self.neighbors[position]

            greedy_policy = max(neighbors,
                                key=lambda x: self.q_values[(position, x)]())

            self.policy[position] = greedy_policy

        return gamma_weigted_return

    def get_state_values(self):
        """Retunrs a Dictionary with every state and it's Markov Policy Value"""

        states = {cords: aggregator()
                  for cords, aggregator in self.state_values.items()}

        return states

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

    def animate_episode(
        self, title="Initial Gridworld", values=None, start_point=(0, 0), nbuffer=10
    ):

        fig, image = self.base_viz(values, legend_small=True)
        self.plot_arrows()

        ax = plt.gca()
        ttl = plt.text(
            0.5,
            1.05,
            f"Current Reward: {0:.2f}",
            transform=ax.transAxes,
            ha="center",
            # va="center",
            fontsize=17,
        )

        (agent,) = plt.plot(*self.start, "-ro", markersize=25)

        while True:

            history = self.create_episode(self.start)

            text = f"History is {len(history)}. Still animate? - y/n/r\n"

            t = input(text)

            while t not in ["y", "n", "r"] and not t.isnumeric():
                print(t)
                t = input(text)

            if "y" == t:
                break
            elif "n" == t:
                exit()
            elif t.isnumeric():
                history[:int(t)]
                break

        states = [i[0] for i in history]
        rewards = [i[1] for i in history]

        state = states[-1]
        action = self.policy[state]

        reward, last_state,  = self.sample_from_environment((state, action))

        states.append(last_state)
        rewards.append(reward)

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

        steps = np.arange(0, (len(states) - 1) * nbuffer)

        ani = animation.FuncAnimation(fig, update, steps)

        return ani

    def sample_counts(self, title="Number of Samples per Tile", savefig=False):

        image = np.zeros(self.dimensions[::-1]) + 0.1

        for (x, y), agg in self.state_values.items():

            image[y, x] = agg.counter

        dtype = np.array(["10hallosa"]).dtype

        labels = np.full(image.shape, "", dtype=dtype)

        mask = image < 100

        labels[mask] = image[mask].astype(dtype)

        for y, x in np.ndindex(labels.shape):

            val = labels[y, x]

            if "." in val:
                labels[y, x] = val.replace(".", "")

            if (x, y) == self.start:
                labels[y, x] = "Start"

            if (x, y) in self.walls:
                labels[y, x] = "Wall"

            if (x, y) == self.goal:
                labels[y, x] = "Goal"

            if (x, y) in self.traps:
                labels[y, x] = "Trap"

        plt.figure(
            figsize=(self.dimensions[0] * 1.5, self.dimensions[1] * 1.2))

        sns.heatmap(image, annot=labels.copy(),
                    cmap=OranbeBlue,  norm=LogNorm(), fmt="", cbar_kws={'label': 'N Samples'})

        plt.title(title, fontsize=17)
        if savefig:
            plt.savefig(
                savefig, dpi=400, bbox_inches="tight")

        plt.show()

    def animate_values(self, FREQ=20, N=7000):

        vals = [self.get_state_values().copy()]
        iterations = [0]

        for i in tqdm.tqdm(range(N), desc="Doing Policy Iterations"):
            self.step()

            if i % FREQ == 0:

                vals.append(self.get_state_values().copy())
                iterations.append(i)

        vmin = min(map(lambda x: min(x.values()), vals))
        vmax = max(map(lambda x: max(x.values()), vals))

        image = np.zeros(self.dimensions[::-1]) + vmin

        frames = []

        for state_values, step in zip(vals, iterations):

            for (x, y), v in state_values.items():

                image[y, x] = v

            frames.append((image.copy(), step))

        image = frames[0][0]

        fig = plt.figure(
            figsize=(self.dimensions[0] * 1.5, self.dimensions[1] * 1.2))

        ax = plt.gca()

        ax1_divider = make_axes_locatable(ax)
        im = plt.imshow(image, OranbeBlue, vmin=vmin, vmax=vmax)

        ttl = plt.text(
            0.5,
            1.05,
            f"State Values for Step {0}",
            transform=ax.transAxes,
            ha="center",
            # va="center",
            fontsize=17,
        )

        for y, x in np.ndindex(image.shape):

            text = None

            if (x, y) == self.start:
                text = "Start"

            if (x, y) in self.walls:
                text = "Wall"

            if (x, y) == self.goal:
                text = "Goal"

            if (x, y) in self.traps:
                text = "Trap"

            if text is not None:
                fontcolor = 'black' if image[y, x] > vmin else "white"

                plt.text(x, y, text, ha="center", va="center",
                         size=14, color=fontcolor)

        cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")
        cb1 = plt.colorbar(im, cax=cax1)
        cax1.set_title('State Values', fontsize=12)

        def update(tup):

            frame, i = tup

            im.set_array(frame)

            ttl.set_text(
                f"State Values for Step {i:4d}"
            )

            return im, ttl

        ani = animation.FuncAnimation(fig, update, frames)

        return ani
