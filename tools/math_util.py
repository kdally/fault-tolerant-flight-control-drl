import numpy as np


def safe_mean(arr):
    """
    Compute the mean of an array if there is at least one element.
    For empty array, return nan. It is used for logging only.

    :param arr: (np.ndarray)
    :return: (float)
    """
    return np.nan if len(arr) == 0 else np.mean(arr)


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
    :param action: (np.ndarray)
    :return: (np.ndarray)
    """
    low, high = action_space.low, action_space.high
    return low + (0.5 * (scaled_action + 1.0) * (high - low))
