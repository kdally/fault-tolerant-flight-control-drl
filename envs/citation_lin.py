import os

import gym
import matlab.engine
import numpy as np


def d2r(num):
    return num * np.pi / 180.0


def r2d(num):
    return num * 180 / np.pi


def map_to(num, a, b):
    """ Map linearly num on the [-1, 1] range to the [a, b] range"""
    return ((num + 1) / 2) * (b - a) + a


class Citation(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['graph']}

    def __init__(self, time_vector: np.ndarray = np.arange(0, 30, 0.01), task=None):

        super(Citation, self).__init__()

        self.time = time_vector
        self.dt = self.time[1] - self.time[0]
        self.A_matrix, self.B_matrix = self.get_eom()

        # Integration step from EOM using Euler Integration
        self.euler = lambda x, u: x + (self.A_matrix.dot(x) + self.B_matrix.dot(u)) * self.dt

        if task is None:
            task = self.get_task_default()
        self.ref_signal = task[0]
        self.track_indices = task[1]
        self.obs_indices = task[2]
        self.observation_space = gym.spaces.Box(-3000, 3000, shape=(4,), dtype=np.float64)
        self.action_space = gym.spaces.Box(-1., 1., shape=(3,), dtype=np.float64)

        self.state = None
        self.scale_s = None
        self.state_history = None
        self.action_history = None
        self.error = None
        self.step_count = None

    def step(self, action: np.ndarray):

        self.state = self.euler(self.state, self.scale_a(action))

        self.error = d2r(self.ref_signal[:, self.step_count]) - self.state[self.track_indices]
        if 5 in self.track_indices:   # for sideslip angle
            self.error[self.track_indices.index(5)] *= 100

        self.state_history[:, self.step_count] = np.multiply(self.state, self.scale_s)
        self.action_history[:, self.step_count] = self.scale_a(action, to='fig')

        self.step_count += 1
        done = bool(self.step_count >= self.time.shape[0])

        return self.get_obs(), self.get_reward(), done, {}

    def reset(self):

        self.state = np.zeros(12)
        self.scale_s = np.ones(self.state.shape)
        self.scale_s[[0, 1, 2, 4, 5, 6, 7, 8]] = 180 / np.pi
        self.state_history = np.zeros((self.state.shape[0], self.time.shape[0]))
        self.action_history = np.zeros((self.action_space.shape[0], self.time.shape[0]))
        self.error = np.zeros(len(self.track_indices))
        self.step_count = 0
        return np.zeros(self.observation_space.shape)

    def get_reward(self):

        sum_error = r2d(self.error.sum() / 30)
        return -abs(max(min(sum_error, 1), -1))

    def get_task_default(self):

        ref_pbody = np.hstack([5 * np.sin(self.time[:int(self.time.shape[0] / 3)] * 2 * np.pi * 0.2),
                               5 * np.sin(self.time[:int(self.time.shape[0] / 3)] * 3.5 * np.pi * 0.2),
                               - 5 * np.ones(int(2.5 * self.time.shape[0] / self.time[-1].round())),
                               5 * np.ones(int(2.5 * self.time.shape[0] / self.time[-1].round())),
                               np.zeros(int(5 * self.time.shape[0] / self.time[-1].round())),
                               ])
        ref_qbody = np.hstack([5 * np.sin(self.time[:int(self.time.shape[0] / 3)] * 2 * np.pi * 0.2),
                               5 * np.sin(self.time[:int(self.time.shape[0] / 3)] * 3.5 * np.pi * 0.2),
                               - 5 * np.ones(int(2.5 * self.time.shape[0] / self.time[-1].round())),
                               5 * np.ones(int(2.5 * self.time.shape[0] / self.time[-1].round())),
                               np.zeros(int(5 * self.time.shape[0] / self.time[-1].round())),
                               ])
        ref_beta = np.zeros(int(self.time.shape[0]))

        return np.vstack([ref_pbody, ref_qbody, ref_beta]), np.array([0, 1, 5])

    def get_obs(self):
        return np.hstack([self.error, self.state[2]])

    @staticmethod
    def scale_a(action: np.ndarray, to: str = 'model'):
        """Min-max un-normalization from [-1, 1] action space to actuator limits"""

        if np.greater(np.abs(action), 1).any():
            print(f'Control input {np.abs(action).max()} is outside [-1, 1] bounds. Corrected to {max(min(np.abs(action).max(), 1), -1)}.')
            action[0] = max(min(action[0], 1), -1)
            action[1] = max(min(action[1], 1), -1)
            # raise Exception(f'Control input {np.abs(action).max()} is outside [-1, 1] bounds.')

        action[0] = map_to(action[0], d2r(-20.05), d2r(14.90))
        action[1] = map_to(action[1], d2r(-37.24), d2r(37.24))
        action[2] = map_to(action[2], d2r(-21.77), d2r(21.77))

        if to == 'model':
            return action
        else:
            return r2d(action)

    @staticmethod
    def get_eom():

        eng = matlab.engine.start_matlab()
        try:
            eng.cd('envs')
        except matlab.engine.MatlabExecutionError:
            pass
        eng.save_mat(nargout=0)
        A = np.asarray(eng.eval('Alin'))
        B = np.asarray(eng.eval('Blin'))[:, :3]
        eng.quit()

        return A, B

    def render(self, mode='any'):
        raise NotImplementedError()

    def close(self):
        pass


# from stable_baselines.common.env_checker import check_env
#
# envs = Citation()
#
# # Box(4,) means that it is a Vector with 4 components
# print("Observation space:", envs.observation_space)
# print("Action space:", envs.action_space)
#
# check_env(envs, warn=True)
