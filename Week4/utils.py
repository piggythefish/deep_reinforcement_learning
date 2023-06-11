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
