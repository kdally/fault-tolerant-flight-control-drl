import gym
import numpy as np
from abc import ABC, abstractmethod
from agent.sac import SAC
from tools.get_task import choose_task
from tools.plot_response import plot_response
import importlib


def d2r(num):
    return num * np.pi / 180.0


def r2d(num):
    return num * 180 / np.pi


def map_to(num: np.ndarray, a, b):
    """ Map linearly num on the [-1, 1] range to the [a, b] range"""
    return ((num + 1.0) / 2.0) * (b - a) + a


class Citation(gym.Env, ABC):
    """Custom Environment that follows gym interface"""

    def __init__(self, evaluation=False, FDD=False):
        super(Citation, self).__init__()

        self.failure_input, self.C_MODEL = self.get_plant()
        self.task_fun, self.failure_input, self.evaluation, self.FDD = choose_task(evaluation, self.failure_input, FDD)

        self.time = self.task_fun()[3]
        self.dt = self.time[1] - self.time[0]
        self.ref_signal = self.task_fun()[0]
        self.track_indices = self.task_fun()[1]
        self.obs_indices = self.task_fun()[2]

        self.sideslip_factor, self.pitch_factor, self.roll_factor = self.adapt_to_failure()

        self.observation_space = gym.spaces.Box(-100, 100, shape=(len(self.obs_indices) + 3,), dtype=np.float64)
        self.action_space = gym.spaces.Box(-1., 1., shape=(3,), dtype=np.float64)
        self.current_deflection = np.zeros(3)

        self.state = None
        self.state_deg = None
        self.scale_s = None
        self.state_history = None
        self.action_history = None
        self.error = None
        self.step_count = None

    def step(self, action_rates: np.ndarray):

        self.current_deflection = self.bound_a(self.current_deflection + self.scale_a(action_rates) * self.dt)
        if self.sideslip_factor[self.step_count - 1] == 0.0: self.current_deflection[2] = 0.0

        if self.time[self.step_count] < 5.0 and self.evaluation:
            self.state = self.C_MODEL.step(
                np.hstack([d2r(self.current_deflection), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self.failure_input[1]]))
        else:
            self.state = self.C_MODEL.step(
                np.hstack([d2r(self.current_deflection), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self.failure_input[2]]))
        self.state_deg = self.state * self.scale_s

        self.error = d2r(self.ref_signal[:, self.step_count] - self.state_deg[self.track_indices])
        self.error[self.track_indices.index(5)] *= self.sideslip_factor[self.step_count]
        self.error[self.track_indices.index(6)] *= self.roll_factor[self.step_count]
        if 7 in self.track_indices:
            self.error[self.track_indices.index(7)] *= self.pitch_factor[self.step_count]
        if 9 in self.track_indices:
            self.error[self.track_indices.index(9)] *= 1.0

        self.state_history[:, self.step_count] = self.state_deg
        self.action_history[:, self.step_count] = self.current_deflection

        self.step_count += 1
        done = bool(self.step_count >= self.time.shape[0])
        if np.isnan(self.state).sum() > 0:
            print(self.state_history[:, self.step_count - 2], self.time[self.step_count - 1])
            exit()
        if self.state[9] <= 50.0 or self.state[9] >= 1e4 or np.greater(np.abs(r2d(self.state[:3])), 1e4).any():
            return np.zeros(self.observation_space.shape), -1 * self.time.shape[0], True, {'is_success': False}

        return self.get_obs(), self.get_reward(), done, {'is_success': True}

    def reset(self):

        self.reset_soft()
        self.ref_signal = self.task_fun()[0]
        return np.zeros(self.observation_space.shape)

    def reset_soft(self):

        self.C_MODEL.initialize()
        action_trim = np.array(
            [-0.024761262011031245, 1.3745996716698875e-14, -7.371050575286063e-14, 0., 0., 0., 0., 0.,
             0.38576210972746433, 0.38576210972746433, self.failure_input[1]])
        self.state = self.C_MODEL.step(action_trim)
        self.scale_s = np.ones(self.state.shape)
        self.scale_s[[0, 1, 2, 4, 5, 6, 7, 8]] = 180 / np.pi
        self.state_deg = self.state * self.scale_s
        self.state_history = np.zeros((self.state.shape[0], self.time.shape[0]))
        self.action_history = np.zeros((self.action_space.shape[0], self.time.shape[0]))
        self.error = np.zeros(len(self.track_indices))
        self.step_count = 0
        self.current_deflection = np.zeros(3)
        return np.zeros(self.observation_space.shape)

    def get_reward(self):

        max_bound = np.ones(self.error.shape)
        reward_vec = np.abs(np.maximum(np.minimum(r2d(self.error / 30), max_bound), -max_bound))
        reward = -reward_vec.sum() / self.error.shape[0]
        return reward

    def get_obs(self):

        untracked_obs_index = np.setdiff1d(self.obs_indices, self.track_indices)
        return np.hstack([self.error, self.state[untracked_obs_index], self.current_deflection])

    @staticmethod
    def scale_a(action_unscaled: np.ndarray) -> np.ndarray:
        """Min-max un-normalization from [-1, 1] action space to actuator limits"""

        max_bound = np.array([15, 40, 20])
        action_scaled = map_to(action_unscaled, -max_bound, max_bound)

        return action_scaled

    @staticmethod
    def bound_a(action):

        min_bounds = np.array([-20.05, -37.24, -21.77])
        max_bounds = np.array([14.9, 37.24, 21.77])
        return np.minimum(np.maximum(action, min_bounds), max_bounds)

    @abstractmethod
    def get_plant(self):
        pass

    @abstractmethod
    def adapt_to_failure(self):
        pass

    def render(self, agent=None, during_training=False, verbose=1):

        if agent is None:
            agent = SAC.load(f"agent/trained/tmp/best_model.zip", env=self)
            agent.save(f'agent/trained/{self.task_fun()[4]}_last.zip')
            agent.ID = 'last'
            assert not self.FDD

        if during_training:
            agent.ID = 'during_training'
            verbose = 0

        if self.FDD:
            agent_robust = agent[0]
            agent_adaptive = agent[1]
        else:
            agent_robust = agent

        obs = self.reset_soft()
        return_a = 0

        for i, current_time in enumerate(self.time):
            if current_time < self.time[-1] / 2 or not self.FDD:
                action, _ = agent_robust.predict(obs, deterministic=True)
            else:
                action, _ = agent_adaptive.predict(obs, deterministic=True)
            obs, reward, done, info = self.step(action)
            return_a += reward
            if current_time == self.time[-1]:
                plot_response(agent.ID, self, self.task_fun(), return_a, during_training,
                              self.failure_input[0], FDD=self.FDD)
                if verbose > 0:
                    print(f"Goal reached! Return = {return_a:.2f}")
                    print('')
                break

    def close(self):
        self.C_MODEL.terminate()
        return


class CitationNormal(Citation):

    def get_plant(self):

        plant = importlib.import_module(f'envs.ice._citation', package=None)
        return plant, ['normal', 1.0, 1.0]

    def adapt_to_failure(self):

        pitch_factor = np.ones(self.time.shape[0])
        roll_factor = np.ones(self.time.shape[0])
        if self.evaluation:
            sideslip_factor = 4.0 * np.ones(self.time.shape[0])
            if self.task_fun()[4] == 'altitude_2attitude':
                roll_factor = 2 * np.ones(self.time.shape[0])
        else:
            sideslip_factor = 10.0 * np.ones(self.time.shape[0])

        return sideslip_factor, pitch_factor, roll_factor


class CitationRudderStuck(Citation):

    def get_plant(self):

        plant = importlib.import_module(f'envs.dr._citation', package=None)
        return plant, ['dr', 0.0, -15.0]

    def adapt_to_failure(self):

        pitch_factor = np.ones(self.time.shape[0])
        roll_factor = np.ones(self.time.shape[0])
        if self.task_fun()[4] == 'altitude_2attitude' and self.evaluation:
            roll_factor = 2 * np.ones(self.time.shape[0])

        sideslip_factor = np.zeros(self.time.shape[0])
        if self.FDD:
            sideslip_factor[:int(self.time.shape[0] / 2)] = 4.0 * np.ones(int(self.time.shape[0] / 2))

        return sideslip_factor, pitch_factor, roll_factor


class CitationIcing(Citation):

    def get_plant(self):

        plant = importlib.import_module(f'envs.ice._citation', package=None)
        return plant, ['ice', 1.0, 0.7]

    def reset(self):

        self.reset_soft()
        self.ref_signal = self.task_fun(theta_angle=25)[0]
        return np.zeros(self.observation_space.shape)

    def adapt_to_failure(self):

        pitch_factor = np.ones(self.time.shape[0])
        roll_factor = np.ones(self.time.shape[0])
        if self.evaluation:
            sideslip_factor = 4.0 * np.ones(self.time.shape[0])
            if self.task_fun()[4] == 'altitude_2attitude':
                roll_factor = 2 * np.ones(self.time.shape[0])
        else:
            sideslip_factor = 10.0 * np.ones(self.time.shape[0])
        self.ref_signal = self.task_fun(theta_angle=25)[0]

        return sideslip_factor, pitch_factor, roll_factor

# from stable_baselines.common.env_checker import check_env
#
# envs = Citation()
#
# # Box(4,) means that it is a Vector with 4 components
# print("Observation space:", envs.observation_space.shape)
# print("Action space:", envs.action_space)
#
# check_env(envs, warn=True)
