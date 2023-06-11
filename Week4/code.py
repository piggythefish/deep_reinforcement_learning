import gymnasium as gym
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
import tqdm


LOSS = tf.keras.losses.Huber()
CNN_SHAPE = (84, 84)


def triple_conv_block_no_batchnorm(x, filters):

    x = tf.keras.layers.Conv2D(
        filters, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(
        filters, 3, padding='same', activation='relu')(x) + x
    x = tf.keras.layers.Conv2D(
        filters, 3, padding='same', activation='relu')(x) + x

    return x


def get_standard_dqn():
    tf.keras.backend.clear_session()

    inputs = tf.keras.layers.Input(CNN_SHAPE + (3,))

    x = triple_conv_block_no_batchnorm(inputs, 16)
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = triple_conv_block_no_batchnorm(x, 32)
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = triple_conv_block_no_batchnorm(x, 64)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(512, "relu")(x)
    outputs = tf.keras.layers.Dense(4, "linear")(x)

    model = tf.keras.Model(inputs, outputs, name="standard_dqn")

    return model


def get_small_dqn():
    tf.keras.backend.clear_session()

    inputs = tf.keras.layers.Input(CNN_SHAPE + (3,))
    x = triple_conv_block_no_batchnorm(inputs, 10)
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = triple_conv_block_no_batchnorm(x, 20)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(512, "relu")(x)
    outputs = tf.keras.layers.Dense(4, "linear")(x)

    model = tf.keras.Model(inputs, outputs, name="standard_dqn")

    return model


@tf.function
def sample_trajectory(dqn, state, epsilon=0.2):

    n_par = tf.shape(state)[0]

    mask = tf.random.uniform((n_par,), 0, 1, tf.float32) > epsilon

    predictions = dqn(state, training=False)
    max_actions = tf.math.argmax(predictions, axis=-1)

    random_choices = tf.random.uniform(
        shape=[n_par], minval=0, maxval=4, dtype=tf.int64)

    return tf.where(mask, max_actions, random_choices), tf.reduce_max(predictions, -1)


@tf.function
def preprocess_all(observation, next_observation, action, reward, terminated):

    observation = tf.cast(observation, tf.float32)/255
    observation = tf.image.resize(observation, CNN_SHAPE)

    next_observation = tf.cast(next_observation, tf.float32)/255
    next_observation = tf.image.resize(next_observation, CNN_SHAPE)

    action = tf.cast(action, tf.int64)
    reward = tf.cast(reward, tf.float32)
    terminated = tf.cast(terminated, tf.bool)

    return observation, next_observation, action, reward, terminated


@tf.function
def preprocess_obersvation(observation):

    observation = tf.cast(observation, tf.float32)/255

    return tf.image.resize(observation, CNN_SHAPE)


@tf.function
def polyak_averaging(Q_target, Q_dqn, tau):
    """

    Args:
        Q_target (_type_): _description_
        Q_dqn (_type_): _description_
        tau (_type_): _description_
    """

    for old, new in zip(Q_target.trainable_variables, Q_dqn.trainable_variables):
        update = old * (1 - tau) + new * tau
        old.assign(update)


@tf.function
def update_q_network(data, dqn, q_target, optimizer, gamma):

    state, next_state, action, reward, terminated = data

    s_prime_values = q_target(next_state, training=False)
    s_prime_values = tf.reduce_max(s_prime_values, -1)
    mask = 1 - tf.cast(terminated, tf.float32)

    labels = reward + gamma * mask * s_prime_values

    with tf.GradientTape() as tape:

        predictions = dqn(state, training=True)
        action_values = tf.gather(predictions, action, batch_dims=1)

        loss = LOSS(action_values, labels)

    gradients = tape.gradient(loss, dqn.trainable_variables)
    optimizer.apply_gradients(zip(gradients, dqn.trainable_variables))
    return loss


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
