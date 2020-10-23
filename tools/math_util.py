import numpy as np
import tensorflow as tf
import gym
import random


def d2r(num):
    return num * np.pi / 180.0


def r2d(num):
    return num * 180 / np.pi


def scale_action(action_space, action):
    """
    Rescale the action from [low, high] to [-1, 1]
    (no need for symmetric action space)

    :param action_space: (gym.spaces.box.Box)
    :param action: (np.ndarray)
    :return: (np.ndarray)
    """
    low, high = action_space.low, action_space.high
    return 2.0 * ((action - low) / (high - low)) - 1.0


def unscale_action(action_space, scaled_action):
    """
    Rescale the action from [-1, 1] to [low, high]
    (no need for symmetric action space)

    :param action_space: (gym.spaces.box.Box)
    :param scaled_action: (np.ndarray)
    :return: (np.ndarray)
    """
    low, high = action_space.low, action_space.high
    return low + (0.5 * (scaled_action + 1.0) * (high - low))


def set_global_seeds(seed):
    """
    set the seed for python random, tensorflow, numpy and gym spaces
    :param seed: (int) the seed
    """
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # prng was removed in latest gym version
    if hasattr(gym.spaces, 'prng'):
        gym.spaces.prng.seed(seed)