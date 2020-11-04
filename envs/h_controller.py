import gym
import numpy as np
from abc import ABC, abstractmethod
from agent2.sac import SAC as SAC2
from tools.get_task import CascadedAltTask
from tools.plot_response import plot_response
import importlib
from tools.math_util import unscale_action, d2r, r2d
from envs.citation import CitationNormal


class AltController(gym.Env, ABC):
    """Custom Environment that follows gym interface"""

    def __init__(self, evaluation=False, InnerAgent='9VZ5VE'):
        super(AltController, self).__init__()

        self.InnerController = CitationNormal(evaluation=evaluation, task=CascadedAltTask)
        self.InnerAgent = SAC2.load(f"agent/trained/3attitude_step_{InnerAgent}.zip", env=self.InnerController)
        self.pitch_limits = self.ActionLimits(np.array([[-30], [30]]))
        self.time = self.InnerController.time
        self.dt = self.InnerController.dt
        self.task_fun = self.InnerController.task_fun
        self.ref_signal = self.InnerController.external_ref_signal = self.task_fun()[5]
        self.obs_indices = self.task_fun()[6]
        self.track_index = self.task_fun()[7]

        self.observation_space = gym.spaces.Box(-100, 100, shape=(len(self.obs_indices) + 3,), dtype=np.float64)
        self.action_space = gym.spaces.Box(-1., 1., shape=(1,), dtype=np.float64)

        self.obs_inner_controller = None
        self.state = None
        self.action_history = None
        self.error = None
        self.step_count = None

    def step(self, pitch_ref: np.ndarray):

        self.step_count = self.InnerController.step_count
        self.InnerController.ref_signal[2, self.step_count] = self.scale_a(self.bound_a(pitch_ref))
        action, _ = self.InnerAgent.predict(self.obs_inner_controller, deterministic=True)
        self.obs_inner_controller, _, done, info = self.InnerController.step(action)
        self.error = self.ref_signal[self.step_count] - self.InnerController.state[self.track_index]

        return self.get_obs(), self.get_reward(), done, info

    def reset(self):

        self.action_history = np.zeros((self.action_space.shape[0], self.time.shape[0]))
        self.error = np.zeros(1)
        self.obs_inner_controller = self.InnerController.reset()
        return np.hstack([self.obs_inner_controller, 0.0])

    def get_reward(self):

        max_bound = np.ones(self.error.shape)
        reward = np.abs(np.maximum(np.minimum(r2d(self.error / 30), max_bound), -max_bound))
        return reward

    def get_obs(self):
        return np.hstack([self.error, self.InnerController.get_obs()])

    def scale_a(self, action_unscaled: np.ndarray) -> np.ndarray:
        """Min-max un-normalization from [-1, 1] action space to actuator limits"""

        return unscale_action(self.pitch_limits, action_unscaled)

    def bound_a(self, action):

        return np.minimum(np.maximum(action, -1), 1)

    def render(self, agent=None, during_training=False, verbose=1):
        self.InnerController.render(agent, during_training, verbose)

    def close(self):
        self.InnerController.close()
        return

    class ActionLimits:

        def __init__(self, limits):
            self.low, self.high = limits[0, :], limits[1, :]


# from stable_baselines.common.env_checker import check_env
#
# envs = AltController()
#
# # Box(4,) means that it is a Vector with 4 components
# print("Observation space:", envs.observation_space.shape)
# print("Action space:", envs.action_space)
#
# check_env(envs, warn=True)
