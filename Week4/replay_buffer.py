import random
import numpy as np
import tensorflow as tf
from utils import preprocess_all, preprocess_obersvation


class ReplayBuffer:
    """
    Class for managing a replay buffer for reinforcement learning.
    """

    def __init__(self, ) -> None:
        """
        Initialize the ReplayBuffer instance.

        Args:
            preprocess_func: Function to preprocess examples.
        """
        self.saved_trajectories = []

    def add_new_trajectory(self, trajectory):
        """
        Add a new trajectory to the replay buffer.

        Args:
            trajectory: List of examples representing a trajectory.
        """
        self.saved_trajectories.append(trajectory)

    def drop_first_trajectory(self):
        """
        Remove the oldest trajectory from the replay buffer.
        """
        self.saved_trajectories.pop(0)

    def sample_singe_example(
        self,
    ):
        """
        Sample a single example from the replay buffer.

        Args:
            melt_stop_criteria: Boolean flag indicating whether to consider stop criteria (default: False).

        Returns:
            example: A single example from a randomly chosen trajectory.
        """
        trajectory = random.choice(self.saved_trajectories)
        example = random.choice(trajectory)

        states, next_states, actions, rewards, terminations, = example

        return states, next_states, actions, rewards, terminations

    def sample_n_examples(self, n_examples: int):
        """
        Sample multiple examples from the replay buffer.

        Args:
            n_examples: The number of examples to sample.

        Returns:
            states, next_states, actions, rewards, stop_criteria: Arrays of sampled examples.
        """
        trajectories = [self.sample_singe_example() for _ in range(n_examples)]

        states, next_states, actions, rewards, stop_criteria = map(
            np.array, zip(*trajectories)
        )

        return states, next_states, actions, rewards, stop_criteria

    def generate_tf_dataset(self, n_batches, batchsize):
        """
        Generate a TensorFlow dataset from the replay buffer.

        Args:
            n_batches: The number of batches to generate.
            batchsize: The size of each batch.

        Returns:
            ds: TensorFlow dataset containing the preprocessed examples.
        """
        n_steps = n_batches * batchsize

        ds = self.sample_n_examples(n_steps)
        ds = tf.data.Dataset.from_tensor_slices(ds)
        ds = ds.map(preprocess_all, tf.data.AUTOTUNE)
        ds = ds.batch(batchsize)

        return ds
