import gym
import numpy as np
import matplotlib.pyplot as plt
import matlab.engine


class Citation(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['graph']}

    def __init__(self, fig_name: object = 'lin', time_vector: np.ndarray = np.arange(0, 20, 0.01),
                 evaluation: object = False):
        super(Citation, self).__init__()

        self.fig_name = fig_name
        self.A_matrix, self.B_matrix = self.get_eom()

        # Integration step from EOM using Euler Integration
        self.euler = lambda x, u: x + (self.A_matrix.dot(x) + self.B_matrix.dot(u)) * self.dt
        # todo: runge-kutta

        self.time = time_vector
        self.dt = self.time[1] - self.time[0]
        self.max_steps = len(self.time)

        self.ref_signal = self.get_task(evaluation)
        self.observation_space = gym.spaces.Box(-3000, 3000, shape=(self.ref_signal.shape[0],), dtype=np.float64)
        self.action_space = gym.spaces.Box(-1., 1., shape=(3,), dtype=np.float64)

    def step(self, action):

        self.state = self.euler(self.state, self.scale_a(action))
        self.state = self.boundaries(self.state)

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

    def boundaries(self, state):

        # state[1] = self.clip(state[1], self.cart.x_min, self.cart.x_max)
        # state[4] = self.clip(state[4], self.ball.th_min, self.ball.th_max)

        return state

    def close(self):
        pass

    def get_obs(self):

        # s = self.state
        observation = self.error
        return self.scale_o(observation)

    def get_reward(self):

        return -abs(self.clip(self.error.sum(), -1, 1))

    @staticmethod
    def scale_o(obs):  # z-normalization

        # obs[0] = obs[0] /
        # obs[1] = obs[1] /
        # obs[2] = obs[2] /
        # obs[3] = obs[3] /

        return obs

    def scale_a(self, action):  # z-normalization

        action[0] = self.clip(action[0], -20.05, 14.90)
        action[1] = self.clip(action[1], -37.24, 37.24)
        action[2] = self.clip(action[2], -21.77, 21.77)
        return action

    @staticmethod
    def clip(value, low, high):
        return max(min(value, high), low)

    @staticmethod
    def get_eom():

        eng = matlab.engine.start_matlab()
        eng.save_mat(nargout=0)
        A = np.asarray(eng.eval('Alin'))
        B = np.asarray(eng.eval('Blin'))[:, :3]

        return A, B

    def get_task(self, evaluation):

        if evaluation:
            ref_pbody = np.hstack([np.zeros(int(5 * self.max_steps / 20)),
                                   6 * np.ones(int(5 * self.max_steps / 20)),
                                   np.zeros(int(5 * self.max_steps / 20)),
                                   3 * np.ones(int(5 * self.max_steps / 20)),
                                   ])
            ref_qbody = np.hstack([3 * np.ones(int(4 * self.max_steps / 20)),
                                   np.zeros(int(4 * self.max_steps / 20)),
                                   4 * np.ones(int(4 * self.max_steps / 20)),
                                   np.zeros(int(4 * self.max_steps / 20)),
                                   3 * np.ones(int(4 * self.max_steps / 20)),
                                   ])
            ref_rbody = np.hstack([3 * np.ones(int(4 * self.max_steps / 20)),
                                   np.zeros(int(4 * self.max_steps / 20)),
                                   4 * np.ones(int(4 * self.max_steps / 20)),
                                   np.zeros(int(4 * self.max_steps / 20)),
                                   3 * np.ones(int(4 * self.max_steps / 20)),
                                   ])

        else:
            ref_pbody = np.hstack([np.zeros(int(5 * self.max_steps / 20)),
                                   6 * np.ones(int(5 * self.max_steps / 20)),
                                   np.zeros(int(5 * self.max_steps / 20)),
                                   3 * np.ones(int(5 * self.max_steps / 20)),
                                   ])
            ref_qbody = np.hstack([3 * np.ones(int(4 * self.max_steps / 20)),
                                   np.zeros(int(4 * self.max_steps / 20)),
                                   4 * np.ones(int(4 * self.max_steps / 20)),
                                   np.zeros(int(4 * self.max_steps / 20)),
                                   3 * np.ones(int(4 * self.max_steps / 20)),
                                   ])
            ref_rbody = np.hstack([3 * np.ones(int(4 * self.max_steps / 20)),
                                   np.zeros(int(4 * self.max_steps / 20)),
                                   4 * np.ones(int(4 * self.max_steps / 20)),
                                   np.zeros(int(4 * self.max_steps / 20)),
                                   3 * np.ones(int(4 * self.max_steps / 20)),
                                   ])

        return np.vstack([ref_pbody, ref_qbody, ref_rbody])

    def render(self, mode='graph'):

        if mode != 'graph':
            raise NotImplementedError()

        fig = plt.figure()
        ax = plt.gca()
        plt.plot(self.time, self.state_history[1, :].T, label=r'$x_c$ (SAC)', linewidth=2, c='blue')
        plt.plot(self.time, self.ref_cart, '--', label=r'$x_{ref}$', linewidth=2, c='blue')
        # plt.plot(self.time, self.PID_history[0, :].T, label=r'$x_c$ (PID)', linewidth=2, c='blue')
        # plt.plot(self.time, self.state_history[6, :].T, '*-', label=r'$e_x$', linewidth=2, c='blue')
        plt.plot(self.time, self.state_history[4, :].T, label=r'$\theta$ (SAC)', linewidth=2, c='orange')
        plt.plot(self.time, self.ref_ball, '--', label=r'$\theta_{ref}$', linewidth=2, c='orange')
        # plt.plot(self.time, self.PID_history[1, :].T, label=r'$\theta$ (PID)', linewidth=2, c='orange')
        # plt.plot(self.time, self.state_history[5, :].T, '*-', label=r'$e_\theta$', linewidth=2, c='blue')
        # plt.plot(self.time, self.state_history[0, :].T, label=r'$x_b$', linewidth=2)
        # plt.plot(self.time, self.state_history[3, :].T, label=r'$x$', linewidth=2)
        # plt.plot(time, error_cart, '--', label=r'$error$', linewidth=2, c='red')
        plt.xlabel('Time [s]', fontsize=11)
        plt.ylabel('Position [m], [rad]', fontsize=11)
        plt.legend()
        ax.set_xlim(0, self.time[-2])
        ax.set_xticks(np.arange(0, 22, 2))
        # ax.set_xticks(np.arange(0, 22, 2), minor=True)
        # ax.set_yticks(np.arange(0, 101, 20))
        # ax.set_yticks(np.arange(0, 22, 2), minor=True)
        # ax.grid(which='major', alpha=0.5)
        ax.grid(which='major')
        # ax.set_aspect(1.0 / ax.get_data_ratio() * 0.45)
        plt.savefig(f'models_backup/{self.fig_name}.eps', format='eps')
        plt.show()

        return


from stable_baselines.common.env_checker import check_env

time = np.arange(0, 20, 0.01)
env = Citation(time)

# Box(4,) means that it is a Vector with 4 components
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)

check_env(env, warn=True)
