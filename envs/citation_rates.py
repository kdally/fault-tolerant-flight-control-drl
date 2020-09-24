import gym
import numpy as np

# # Import the low-level C/C++ module
if __package__ or "." in __name__:
    from . import _citation as C_MODEL
else:
    import _citation as C_MODEL


def d2r(num):
    return num * np.pi / 180.0


def r2d(num):
    return num * 180 / np.pi


def map_to(num: float, a, b):
    """ Map linearly num on the [-1, 1] range to the [a, b] range"""
    return ((num + 1.0) / 2.0) * (b - a) + a


class Citation(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['graph']}

    def __init__(self, time_vector: np.ndarray = np.arange(0, 30, 0.01), task=None):

        super(Citation, self).__init__()

        self.time = time_vector
        self.dt = self.time[1] - self.time[0]

        if task is None:
            task = self.get_task_default()
        self.task_fun = task
        self.ref_signal = self.task_fun[0]
        self.track_indices = self.task_fun[1]
        self.obs_indices = self.task_fun[2]
        self.observation_space = gym.spaces.Box(-100, 100, shape=(len(self.obs_indices) + 3,), dtype=np.float64)
        self.action_space = gym.spaces.Box(-1., 1., shape=(3,), dtype=np.float64)
        self.current_deflection = np.zeros(3)

        self.state = None
        self.scale_s = None
        self.state_history = None
        self.action_history = None
        self.error = None
        self.step_count = None

    def step(self, action_rates: np.ndarray):

        self.current_deflection = self.current_deflection + self.scale_a(action_rates)*self.dt
        self.state = C_MODEL.step(np.hstack([d2r(self.current_deflection), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        if np.isnan(self.state).sum() > 0:
            print(self.state)
            raise Exception('Nan')

        self.error = d2r(self.ref_signal[:, self.step_count]) - self.state[self.track_indices]
        if 5 in self.track_indices:  # for sideslip angle, change reward scale due to dimensions difference
            self.error[self.track_indices.index(5)] *= 10

        self.state_history[:, self.step_count] = np.multiply(self.state, self.scale_s)
        self.action_history[:, self.step_count] = self.current_deflection

        self.step_count += 1
        done = bool(self.step_count >= self.time.shape[0])

        return self.get_obs(), self.get_reward(), done, {}

    def reset(self):

        C_MODEL.initialize()
        action_trim = np.array(
            [-0.024761262011031245, 1.3745996716698875e-14, -7.371050575286063e-14, 0., 0., 0., 0., 0.,
             0.38576210972746433, 0.38576210972746433, ])
        self.state = C_MODEL.step(action_trim)
        self.scale_s = np.ones(self.state.shape)
        self.scale_s[[0, 1, 2, 4, 5, 6, 7, 8]] = 180 / np.pi
        self.state_history = np.zeros((self.state.shape[0], self.time.shape[0]))
        self.action_history = np.zeros((self.action_space.shape[0], self.time.shape[0]))
        self.error = np.zeros(len(self.track_indices))
        self.step_count = 0
        self.current_deflection = np.zeros(3)
        self.ref_signal = self.task_fun[0]
        return np.zeros(self.observation_space.shape)

    def get_reward(self):

        reward = 0
        for sig in self.error:
            reward += -abs(max(min(r2d(sig / 30), 1), -1) / self.error.shape[0])
        reward_track = reward

        # action_delta_allow = np.array([1, 2, 1])
        # action_delta = np.abs(self.action_history[:, self.step_count - 1]
        #                       - self.action_history[:, self.step_count - 2])  # step count has already been incremented
        # if (action_delta > action_delta_allow).any():
        #     penalty = np.maximum(np.zeros(3), (action_delta - action_delta_allow))
        #     reward += -penalty.sum() / 200
        #     print(f'Reward for tracking error = {reward_track:.2f}, '
        #           f'penalty = {-penalty.sum() / 200:.2f}')
        # else:
        #     print('safe')

        return reward

    def get_obs(self):

        untracked_obs_index = np.setdiff1d(self.obs_indices, self.track_indices)
        return np.hstack([self.error, self.state[untracked_obs_index], self.current_deflection])

    @staticmethod
    def scale_a(action_unscaled: np.ndarray) -> np.ndarray:
        """Min-max un-normalization from [-1, 1] action space to actuator limits"""

        if np.greater(np.abs(action_unscaled), 1).any():
            print(
                f'Control input {np.abs(action_unscaled).max()} is outside [-1, 1] bounds. '
                f'Corrected to {max(min(np.abs(action_unscaled).max(), 1), -1)}.')
            action_unscaled[0] = max(min(action_unscaled[0], 1), -1)
            action_unscaled[1] = max(min(action_unscaled[1], 1), -1)
            # raise Exception(f'Control input {np.abs(action).max()} is outside [-1, 1] bounds.')

        action_scaled = np.ndarray((3,))
        action_scaled[0] = map_to(action_unscaled[0], d2r(-20), d2r(20))
        action_scaled[1] = map_to(action_unscaled[1], d2r(-40), d2r(40))
        action_scaled[2] = map_to(action_unscaled[2], d2r(-20), d2r(20))

        return r2d(action_scaled)

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

        return np.vstack([ref_pbody, ref_qbody, ref_beta]), [0, 1, 5], [0, 1, 5, 2]

    def render(self, mode='any'):
        raise NotImplementedError()

    def close(self):
        C_MODEL.terminate()
        return

# from stable_baselines.common.env_checker import check_env
#
# envs = Citation()
#
# # Box(4,) means that it is a Vector with 4 components
# print("Observation space:", envs.observation_space)
# print("Action space:", envs.action_space)
#
# check_env(envs, warn=True)
