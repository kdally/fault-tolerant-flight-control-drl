import gym
import numpy as np
from abc import ABC, abstractmethod
from agent.sac import SAC
from tools.get_task import AltitudeTask, AttitudeTask, BodyRateTask
from tools.plot_response import plot_response
import importlib
from tools.math_util import unscale_action, d2r, r2d


class Citation(gym.Env, ABC):
    """Custom Environment that follows gym interface"""

    def __init__(self, evaluation=False, FDD=False, task=AttitudeTask):
        super(Citation, self).__init__()

        self.rate_limits = self.ActionLimits(np.array([[-15, -40, -20], [15, 40, 20]]))
        self.deflection_limits = self.ActionLimits(np.array([[-20.05, -37.24, -21.77], [14.9, 37.24, 21.77]]))
        self.C_MODEL, self.failure_input = self.get_plant()
        self.task = task
        self.task_fun, self.evaluation, self.FDD = self.task().choose_task(evaluation, self.failure_input, FDD)

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
        self.external_ref_signal = None

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
            self.error[self.track_indices.index(9)] *= 0.25

        self.state_history[:, self.step_count] = self.state_deg
        self.action_history[:, self.step_count] = self.current_deflection

        self.step_count += 1
        done = bool(self.step_count >= self.time.shape[0])
        if np.isnan(self.state).sum() > 0:
            print(self.state_history[:, self.step_count - 2], self.time[self.step_count - 1])
            plot_response('before_crash', self, self.task_fun(), 100, during_training=False,
                          failure=self.failure_input[0], FDD=self.FDD, broken=True)
            exit()
        if self.state[9] <= 50.0 or self.state[9] >= 1e4 or np.greater(np.abs(r2d(self.state[:3])), 1e4).any() \
                or np.greater(np.abs(r2d(self.state[6:9])), 1e3).any():
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
        # reward_vec = np.abs(np.maximum(np.minimum(r2d(self.error / 30), max_bound), -max_bound))
        reward_vec = np.abs(r2d(self.error / 30))
        reward = -reward_vec.sum() / self.error.shape[0]
        return reward

    def get_reward_comp(self):
        max_bound = np.ones(self.error.shape)
        # reward_vec = np.abs(np.maximum(np.minimum(r2d(self.error / 30), max_bound), -max_bound))
        reward_vec = np.abs(r2d(self.error / 30))
        return reward_vec

    def get_obs(self):

        untracked_obs_index = np.setdiff1d(self.obs_indices, self.track_indices)
        return np.hstack([self.error, self.state[untracked_obs_index], self.current_deflection])

    def scale_a(self, action_unscaled: np.ndarray) -> np.ndarray:
        """Min-max un-normalization from [-1, 1] action space to actuator limits"""

        return unscale_action(self.rate_limits, action_unscaled)

    def bound_a(self, action):

        return np.minimum(np.maximum(action, self.deflection_limits.low), self.deflection_limits.high)

    def get_cousin(self):
        return self.__init__(evaluation=self.evaluation, FDD=self.FDD, task=self.task)

    @abstractmethod
    def get_plant(self):
        pass

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

    def FFD_change(self):
        pass

    def render(self, agent=None, during_training=False, verbose=1):

        if agent is None:
            agent = SAC.load(f"agent/trained/tmp/best_model.zip", env=self)
            agent.save(f'agent/trained/{self.task_fun()[4]}_last.zip')
            agent.ID = 'last'
            assert not self.FDD, 'Pre-trained agent needs to be passed for Fault Detection and Diagnosis simualtion.'

        if during_training:
            agent.ID = 'during_training'
            verbose = 0

        if self.FDD:
            agent_robust = agent[0]
            agent_adaptive = agent[1]
            agent = agent[1]
        else:
            agent_robust = agent

        obs = self.reset_soft()
        return_a = 0

        for i, current_time in enumerate(self.time):
            if current_time < self.time[-1] / 2 or not self.FDD:
                action, _ = agent_robust.predict(obs, deterministic=True)
            else:
                self.FFD_change()
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

    class ActionLimits:

        def __init__(self, limits):
            self.low, self.high = limits[0, :], limits[1, :]


class CitationNormal(Citation):

    def get_plant(self):

        plant = importlib.import_module(f'envs.normal._citation', package=None)
        return plant, ['normal', 1.0, 1.0]


class CitationRudderStuck(Citation):

    def get_plant(self):

        plant = importlib.import_module(f'envs.dr._citation', package=None)
        return plant, ['dr', 0.0, -9.0]

    def adapt_to_failure(self):

        sideslip_factor, pitch_factor, roll_factor = super(CitationRudderStuck, self).adapt_to_failure()
        sideslip_factor = np.zeros(self.time.shape[0])
        roll_factor = 0.5 * np.ones(self.time.shape[0])
        if self.FDD:
            sideslip_factor[:int(self.time.shape[0] / 2)] = 4.0 * np.ones(int(self.time.shape[0] / 2))
            roll_factor[:int(self.time.shape[0] / 2)] = 2.0 * np.ones(int(self.time.shape[0] / 2))

        return sideslip_factor, pitch_factor, roll_factor


class CitationAileronEff(Citation):

    def get_plant(self):

        plant = importlib.import_module(f'envs.da._citation', package=None)
        return plant, ['da', 1.0, 0.3]

    def adapt_to_failure(self):

        sideslip_factor, _, roll_factor = super(CitationAileronEff, self).adapt_to_failure()
        pitch_factor = 1.5 * np.ones(self.time.shape[0])
        if self.FDD:
            pitch_factor[:int(self.time.shape[0] / 2)] = np.ones(int(self.time.shape[0] / 2))

        return sideslip_factor, pitch_factor, roll_factor


class CitationElevRange(Citation):

    def get_plant(self):

        plant = importlib.import_module(f'envs.de._citation', package=None)
        # self.deflection_limits = self.ActionLimits(np.array([[-3.0, -37.24, -21.77], [3.0, 37.24, 21.77]]))
        # self.rate_limits = self.ActionLimits(np.array([[-7, -40, -20], [7, 40, 20]]))
        return plant, ['de', 20.05, 3.0]

    def FFD_change(self):

        self.deflection_limits = self.ActionLimits(np.array([[-3.0, -37.24, -21.77], [3.0, 37.24, 21.77]]))
        self.rate_limits = self.ActionLimits(np.array([[-7, -40, -20], [7, 40, 20]]))


class CitationCgShift(Citation):

    def get_plant(self):

        plant = importlib.import_module(f'envs.cg._citation', package=None)
        return plant, ['cg', 1.0, 1.04]


class CitationIcing(Citation):

    def get_plant(self):

        plant = importlib.import_module(f'envs.ice._citation', package=None)
        return plant, ['ice', 1.0, 0.7]  # https://doi.org/10.1016/S0376-0421(01)00018-5

    def reset(self):

        super(CitationIcing, self).reset()
        self.ref_signal = self.task_fun(theta_angle=25)[0]
        return np.zeros(self.observation_space.shape)

    def adapt_to_failure(self):

        sideslip_factor, pitch_factor, roll_factor = super(CitationIcing, self).adapt_to_failure()
        self.ref_signal = self.task_fun(theta_angle=25)[0]

        return sideslip_factor, pitch_factor, roll_factor


class CitationHorzTail(Citation):

    def get_plant(self):

        plant = importlib.import_module(f'envs.ht._citation', package=None)
        return plant, ['ht', 1.0, 0.3]


class CitationVertTail(Citation):

    def get_plant(self):

        plant = importlib.import_module(f'envs.vt._citation', package=None)
        return plant, ['vt', 1.0, 0.0]

    def adapt_to_failure(self):

        sideslip_factor, pitch_factor, roll_factor = super(CitationVertTail, self).adapt_to_failure()
        sideslip_factor = 1 * np.ones(self.time.shape[0])
        if self.FDD:
            sideslip_factor[:int(self.time.shape[0] / 2)] = 4.0 * np.ones(int(self.time.shape[0] / 2))

        return sideslip_factor, pitch_factor, roll_factor


class CitationVerif(CitationNormal):

    def step(self, actions: np.ndarray):
        self.current_deflection = actions
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
            plot_response('before_crash', self, self.task_fun(), 100, during_training=False,
                          failure=self.failure_input[0], FDD=self.FDD, broken=True)
            exit()
        if self.state[9] <= 50.0 or self.state[9] >= 1e4 or np.greater(np.abs(r2d(self.state[:3])), 1e4).any() \
                or np.greater(np.abs(r2d(self.state[6:9])), 1e3).any():
            return np.zeros(self.observation_space.shape), -1 * self.time.shape[0], True, {'is_success': False}

        return self.get_obs(), self.get_reward(), done, {'is_success': True}


# from stable_baselines.common.env_checker import check_env
#
# envs = Citation()
#
# # Box(4,) means that it is a Vector with 4 components
# print("Observation space:", envs.observation_space.shape)
# print("Action space:", envs.action_space)
#
# check_env(envs, warn=True)
