import gym
import numpy as np
from tools.get_task import get_task_eval, get_task_tr, get_task_eval_fail, get_task_tr_fail


def d2r(num):
    return num * np.pi / 180.0


def r2d(num):
    return num * 180 / np.pi


def map_to(num: np.ndarray, a, b):
    """ Map linearly num on the [-1, 1] range to the [a, b] range"""
    return ((num + 1.0) / 2.0) * (b - a) + a


class Citation(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['graph']}

    def __init__(self, C_MODEL, eval=False, failure=False):

        super(Citation, self).__init__()
        self.C_MODEL = C_MODEL

        if failure:
            if eval:
                self.task_fun = get_task_eval_fail
            else:
                self.task_fun = get_task_tr_fail
        else:
            if eval:
                self.task_fun = get_task_eval
            else:
                self.task_fun = get_task_tr

        self.time = self.task_fun()[3]
        self.dt = self.time[1] - self.time[0]

        self.ref_signal = self.task_fun()[0]
        self.track_indices = self.task_fun()[1]
        self.obs_indices = self.task_fun()[2]
        self.observation_space = gym.spaces.Box(-100, 100, shape=(len(self.obs_indices) + 3 +1,), dtype=np.float64)
        # self.observation_space = gym.spaces.Box(-100, 100, shape=(len(self.obs_indices) + 3 ,), dtype=np.float64)
        self.action_space = gym.spaces.Box(-1., 1., shape=(3,), dtype=np.float64)
        self.current_deflection = np.zeros(3)

        self.state = None
        self.scale_s = None
        self.state_history = None
        self.action_history = None
        self.error = None
        self.step_count = None

    def step(self, action_rates: np.ndarray):

        self.current_deflection = self.bound_a(self.current_deflection + self.scale_a(action_rates)*self.dt)
        self.state = self.C_MODEL.step(np.hstack([d2r(self.current_deflection), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

        self.error = d2r(self.ref_signal[:, self.step_count]) - self.state[self.track_indices]
        if 5 in self.track_indices:  #  sideslip angle, change reward scale due to dimensions difference
            self.error[self.track_indices.index(5)] *= 4
        self.error[self.track_indices.index(7)] *= 1.2

        self.state_history[:, self.step_count] = self.state*self.scale_s
        self.action_history[:, self.step_count] = self.current_deflection

        self.step_count += 1
        done = bool(self.step_count >= self.time.shape[0])
        if np.isnan(self.state).sum() > 0:
            return np.zeros(self.observation_space.shape), -1, True, {}

        return self.get_obs(), self.get_reward(), done, {}

    def reset_soft(self):

        self.C_MODEL.initialize()
        action_trim = np.array(
            [-0.024761262011031245, 1.3745996716698875e-14, -7.371050575286063e-14, 0., 0., 0., 0., 0.,
             0.38576210972746433, 0.38576210972746433, ])
        self.state = self.C_MODEL.step(action_trim)
        self.scale_s = np.ones(self.state.shape)
        self.scale_s[[0, 1, 2, 4, 5, 6, 7, 8]] = 180 / np.pi
        self.state_history = np.zeros((self.state.shape[0], self.time.shape[0]))
        self.action_history = np.zeros((self.action_space.shape[0], self.time.shape[0]))
        self.error = np.zeros(len(self.track_indices))
        self.step_count = 0
        self.current_deflection = np.zeros(3)
        return np.zeros(self.observation_space.shape)

    def reset(self):

        self.C_MODEL.initialize()
        action_trim = np.array(
            [-0.024761262011031245, 1.3745996716698875e-14, -7.371050575286063e-14, 0., 0., 0., 0., 0.,
             0.38576210972746433, 0.38576210972746433, ])
        self.state = self.C_MODEL.step(action_trim)
        self.scale_s = np.ones(self.state.shape)
        self.scale_s[[0, 1, 2, 4, 5, 6, 7, 8]] = 180 / np.pi
        self.state_history = np.zeros((self.state.shape[0], self.time.shape[0]))
        self.action_history = np.zeros((self.action_space.shape[0], self.time.shape[0]))
        self.error = np.zeros(len(self.track_indices))
        self.step_count = 0
        self.current_deflection = np.zeros(3)
        self.ref_signal = self.task_fun()[0]
        return np.zeros(self.observation_space.shape)

    def get_reward(self):

        max_bound = np.ones(self.error.shape)
        reward_vec = np.abs(np.maximum(np.minimum(r2d(self.error / 30), max_bound), -max_bound))
        reward = -reward_vec.sum() / self.error.shape[0]
        # reward = -reward_vec[:2].sum() / (self.error.shape[0]-1)
        return reward

    def get_obs(self):

        untracked_obs_index = np.setdiff1d(self.obs_indices, self.track_indices)
        # print(np.hstack([self.error, self.state[6]/10, self.state[7]/10, self.state[untracked_obs_index], d2r(self.current_deflection)]))
        return np.hstack([self.error, self.state[6]/5, self.state[untracked_obs_index], self.current_deflection/5])
        # return np.hstack([self.error, self.state[untracked_obs_index], self.current_deflection])

    @staticmethod
    def scale_a(action_unscaled: np.ndarray) -> np.ndarray:
        """Min-max un-normalization from [-1, 1] action space to actuator limits"""

        max_bound = np.array([20, 40, 20])
        action_scaled = map_to(action_unscaled, -max_bound, max_bound)

        return action_scaled

    @staticmethod
    def bound_a(action):

        min_bounds = np.array([-20.05, -37.24, -21.77])
        max_bounds = np.array([14.9, 37.24, 21.77])
        return np.minimum(np.maximum(action, min_bounds), max_bounds)

    def render(self, mode='any'):
        raise NotImplementedError()

    def close(self):
        self.C_MODEL.terminate()
        return

# from stable_baselines.common.env_checker import check_env
#
# envs = Citation()
#
# # Box(4,) means that it is a Vector with 4 components
# print("Observation space:", envs.observation_space.shape)
# print("Action space:", envs.action_space)
#
# check_env(envs, warn=True)
