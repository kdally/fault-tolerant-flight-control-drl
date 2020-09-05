import gym
import matlab.engine
import numpy as np


class Citation(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['graph']}

    def __init__(self, time_vector: np.ndarray = np.arange(0, 20, 0.01), evaluation: bool = False):

        super(Citation, self).__init__()

        self.A_matrix, self.B_matrix = self.get_eom()

        # Integration step from EOM using Euler Integration
        self.euler = lambda x, u: x + (self.A_matrix.dot(x) + self.B_matrix.dot(u)) * self.dt

        self.time = time_vector
        self.dt = self.time[1] - self.time[0]
        self.max_steps = len(self.time)

        self.ref_signal = self.get_task(evaluation)
        self.observation_space = gym.spaces.Box(-3000, 3000, shape=(self.ref_signal.shape[0],), dtype=np.float64)
        self.action_space = gym.spaces.Box(-1., 1., shape=(3,), dtype=np.float64)

        self.state = None
        self.state_history = None
        self.action_history = None
        self.error = None
        self.step_count = None

    def step(self, action: np.ndarray):

        self.state = self.euler(self.state, self.scale_a(action))

        self.error = self.ref_signal[:, self.step_count] - self.state[:self.ref_signal.shape[0]]
        self.state_history[:, self.step_count] = self.state
        self.action_history[:, self.step_count] = self.scale_a(action)

        self.step_count += 1
        done = bool(self.step_count >= self.max_steps)

        return self.get_obs(), self.get_reward(), done, {}

    def reset(self):

        self.state = np.zeros(12)
        self.state_history = np.zeros((self.state.shape[0], self.max_steps))
        self.action_history = np.zeros((self.action_space.shape[0], self.max_steps))
        self.error = np.zeros(self.ref_signal.shape[0])
        self.step_count = 0
        return np.zeros(self.observation_space.shape)

    def get_reward(self):

        return -abs(self.clip(self.error.sum()/60, -1, 1))

    def get_task(self, evaluation):

        if evaluation:
            ref_pbody = np.hstack([np.zeros(int(5 * self.max_steps / 20)),
                                   2 * np.ones(int(5 * self.max_steps / 20)),
                                   np.zeros(int(5 * self.max_steps / 20)),
                                   -2 * np.ones(int(5 * self.max_steps / 20)),
                                   ])
            ref_qbody = np.hstack([3 * np.ones(int(4 * self.max_steps / 20)),
                                   np.zeros(int(4 * self.max_steps / 20)),
                                   -3 * np.ones(int(4 * self.max_steps / 20)),
                                   np.zeros(int(4 * self.max_steps / 20)),
                                   3 * np.ones(int(4 * self.max_steps / 20)),
                                   ])
            ref_beta = np.zeros(int(self.max_steps))

        else:
            ref_pbody = np.hstack([np.zeros(int(5 * self.max_steps / 20)),
                                   2 * np.ones(int(5 * self.max_steps / 20)),
                                   np.zeros(int(5 * self.max_steps / 20)),
                                   -2 * np.ones(int(5 * self.max_steps / 20)),
                                   ])
            ref_qbody = np.hstack([3 * np.ones(int(4 * self.max_steps / 20)),
                                   np.zeros(int(4 * self.max_steps / 20)),
                                   -3 * np.ones(int(4 * self.max_steps / 20)),
                                   np.zeros(int(4 * self.max_steps / 20)),
                                   3 * np.ones(int(4 * self.max_steps / 20)),
                                   ])
            ref_rbody = np.hstack([1 * np.ones(int(4 * self.max_steps / 20)),
                                   np.zeros(int(4 * self.max_steps / 20)),
                                   -1 * np.ones(int(4 * self.max_steps / 20)),
                                   np.zeros(int(4 * self.max_steps / 20)),
                                   1 * np.ones(int(4 * self.max_steps / 20)),
                                   ])
            ref_beta = np.zeros(int(self.max_steps))

        return np.vstack([ref_pbody, ref_qbody, ref_rbody])

    def get_obs(self):

        # s = self.state
        observation = self.error
        return self.scale_o(observation)

    def scale_a(self, action):  # z-normalization

        action[0] = self.clip(action[0], -20.05, 14.90)
        action[1] = self.clip(action[1], -37.24, 37.24)
        action[2] = self.clip(action[2], -21.77, 21.77)
        return action

    @staticmethod
    def scale_o(obs):  # z-normalization

        # obs[0] = obs[0] /
        # obs[1] = obs[1] /
        # obs[2] = obs[2] /
        # obs[3] = obs[3] /

        return obs

    @staticmethod
    def clip(value, low, high):
        return max(min(value, high), low)

    @staticmethod
    def get_eom():

        eng = matlab.engine.start_matlab()
        eng.cd('envs/lin')
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
# time = np.arange(0, 20, 0.01)
# envs = Citation(time)
#
# # Box(4,) means that it is a Vector with 4 components
# print("Observation space:", envs.observation_space)
# print("Action space:", envs.action_space)
#
# check_env(envs, warn=True)
