import gymnasium as gym
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
import tqdm
from utils import sample_trajectory, preprocess_obersvation


class ENV_SAMPLER:
    """
    Class for sampling environment data using a DQN model.
    """

    def __init__(self, dqn, n_multi_envs) -> None:
        """
        Initialize the ENV_SAMPLER instance.

        Args:
            env: The environment to sample from.
            dqn: The DQN model for action selection.
            n_multi_envs: The number of parallel environments.
            preprocess_observation: Function to preprocess observations.
        """
        self.env = gym.vector.make('ALE/Breakout-v5', num_envs=n_multi_envs)
        self.current_state = self.env.reset()[0]
        self.dqn = dqn
        self.n_multi_envs = n_multi_envs

    def reset_env(self):
        """
        Reset the environment to the initial state.
        """
        self.current_state = self.env.reset()[0]

    def sample(self, n_samples, epsilon=0.2):
        """
        Sample environment data.

        Args:
            n_samples: The number of samples to generate.
            epsilon: The exploration factor for action selection (default: 0.2).

        Returns:
            samples: List of sampled data tuples (current_state, next_state, action, reward, terminated).
        """
        samples = []

        n_steps = np.ceil(n_samples / self.n_multi_envs).astype(int)

        for _ in range(n_steps):
            oberservation_as_tensor = preprocess_obersvation(
                self.current_state)

            action, q_vals = map(lambda x: x.numpy(), sample_trajectory(
                self.dqn, oberservation_as_tensor, epsilon))

            observation, reward, terminated, truncated, info = self.env.step(
                action)

            for i in range(self.n_multi_envs):
                samples.append((self.current_state[i],
                                observation[i],
                                action[i],
                                reward[i],
                                terminated[i]))

            self.current_state = observation

        return samples[:n_samples]

    def measure_model_perforamnce(self, gamma: float, target_q):

        self.reset_env()

        rewards = np.zeros(self.n_multi_envs)
        terminated_at = []
        q_values = []
        target_q_values = []

        allready_terminated = np.zeros(self.n_multi_envs, np.bool)

        steps = 0

        while True:

            oberservation_as_tensor = preprocess_obersvation(
                self.current_state)

            action, q_vals = map(lambda x: x.numpy(), sample_trajectory(
                self.dqn, oberservation_as_tensor, 0))

            target_vals = tf.reduce_max(target_q(oberservation_as_tensor), -1)

            observation, reward, terminated, truncated, info = self.env.step(
                action)

            self.current_state = observation

            rewards += (gamma ** steps) * reward * (1 - allready_terminated)

            allready_terminated = np.logical_or(
                allready_terminated, terminated)

            for t in terminated:

                if t:
                    terminated_at.append(steps)

            q_values.extend(q_vals.tolist())
            target_q_values.extend(target_vals.numpy().tolist())

            steps += 1

            if allready_terminated.all():

                break

        average_q_val = np.mean(q_values)
        average_target_q_val = np.mean(target_q_values)

        l2_diff = np.array(q_values) - np.array(target_q_values)
        l2_diff = np.sqrt(np.square(l2_diff).mean())

        average_rewards = np.mean(rewards)
        average_termination = np.mean(terminated_at)

        return average_rewards, average_termination, average_q_val, average_target_q_val, l2_diff
