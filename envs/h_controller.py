import gym
import numpy as np
from abc import ABC, abstractmethod
from agent.sac import SAC
from tools.get_task import CascadedAltTask
from tools.plot_response import plot_response
import importlib
from tools.math_util import unscale_action, d2r, r2d
from envs.citation import CitationNormal


class AltController(gym.Env, ABC):
    """Custom Environment that follows gym interface"""

    def __init__(self, inner_controller=CitationNormal, evaluation=False, FDD=False):
        super(AltController, self).__init__()

        self.InnerController = inner_controller(evaluation=evaluation, task=CascadedAltTask, FDD=FDD)
        self.pitch_limits = self.ActionLimits(np.array([[-30], [30]]))
        self.rate_limits = self.ActionLimits(np.array([[-10], [10]]))
        self.time = self.InnerController.time
        self.dt = self.InnerController.dt
        self.task_fun = self.InnerController.task_fun
        self.ref_signal = self.InnerController.external_ref_signal = self.task_fun()[5]
        self.obs_indices = self.task_fun()[6]
        self.track_index = self.task_fun()[7]

        self.observation_space = gym.spaces.Box(-100, 100, shape=(2,), dtype=np.float64)
        self.action_space = gym.spaces.Box(-1., 1., shape=(1,), dtype=np.float64)

        self.agent, self.agent.ID = self.load_agent()

        self.current_pitch_ref = None
        self.obs_inner_controller = None
        self.state = None
        self.action_history = None
        self.error = None
        self.step_count = None

    def step(self, pitch_ref: np.ndarray):

        self.step_count = self.InnerController.step_count
        self.current_pitch_ref = self.bound_a(self.current_pitch_ref + self.scale_a(pitch_ref) * self.dt)
        self.InnerController.ref_signal[0, self.step_count] = self.current_pitch_ref[0]

        if self.time[self.step_count] < self.InnerController.FDD_switch_time or not self.InnerController.FDD:
            action, _ = self.InnerController.agents[0].predict(self.obs_inner_controller, deterministic=True)
        else:
            self.InnerController.FFD_change()
            action, _ = self.InnerController.agents[1].predict(self.obs_inner_controller, deterministic=True)
        self.obs_inner_controller, _, done, info = self.InnerController.step(action)
        self.error = self.ref_signal[self.step_count] - self.InnerController.state[self.track_index]
        self.error *= 0.25

        return self.get_obs(), self.get_reward(), done, info

    def reset(self):

        self.action_history = np.zeros((self.action_space.shape[0], self.time.shape[0]))
        self.error = np.zeros(1)
        self.current_pitch_ref = np.zeros(1)
        self.obs_inner_controller = self.InnerController.reset()
        self.step_count = self.InnerController.step_count
        return np.hstack([self.error, self.current_pitch_ref])

    def get_reward(self):
        max_bound = np.ones(self.error.shape)
        reward = -np.abs(np.maximum(np.minimum(self.error / 60, max_bound), -max_bound))
        return reward

    def get_obs(self):
        # return np.hstack([self.error, r2d(self.current_pitch_ref)])
        return np.hstack([self.error, self.current_pitch_ref])

    def scale_a(self, action_unscaled: np.ndarray) -> np.ndarray:
        """Min-max un-normalization from [-1, 1] action space to actuator limits"""

        return unscale_action(self.rate_limits, action_unscaled)

    def bound_a(self, action):
        return np.minimum(np.maximum(action, self.pitch_limits.low), self.pitch_limits.high)

    def render(self, ext_agent=None, verbose=1):

        during_training = False
        if ext_agent is not None:
            self.agent = ext_agent
            # self.agent.save(f'agent/trained/{self.task_fun()[4]}_last.zip')
            self.agent.ID = 'last'
            verbose = 0
            during_training = True

        obs_outer_loop = self.reset()
        episode_reward = 0
        done = False
        while not done:

            pitch_ref, _ = self.agent.predict(obs_outer_loop, deterministic=True)
            obs_outer_loop, reward_outer_loop, done, info = self.step(pitch_ref)
            episode_reward += reward_outer_loop

        plot_response(self.agent.ID + '_' + self.InnerController.agentID.split('_')[2], self.InnerController,
                      self.task_fun(), episode_reward, during_training,
                      self.InnerController.failure_input[0], FDD=self.InnerController.FDD)
        if verbose > 0:
            print(f"Goal reached! Return = {episode_reward:.2f}")
            print('')

    def close(self):
        self.InnerController.close()
        return

    def load_agent(self):
        return SAC.load(f"agent/trained/{self.InnerController.task.agent_catalog['normal_outer_loop']}.zip", env=self), \
               self.InnerController.task.agent_catalog['normal_outer_loop']

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
