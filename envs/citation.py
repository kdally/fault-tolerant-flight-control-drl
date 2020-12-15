import gym
import numpy as np
from abc import abstractmethod
from agent.sac import SAC
from tools.get_task import AltitudeTask, AttitudeTask, BodyRateTask, ReliabilityTask
from tools.plot_response import plot_response
import importlib
from tools.math_util import unscale_action, d2r, r2d
from tools.identifier import get_ID
from alive_progress import alive_bar


class Citation(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self, evaluation=False, FDD=False, task=AttitudeTask):
        super(Citation, self).__init__()

        self.rate_limits = self.ActionLimits(np.array([[-20, -40, -20], [20, 40, 20]]))
        self.deflection_limits = self.ActionLimits(np.array([[-20.05, -37.24, -21.77], [14.9, 37.24, 21.77]]))
        self.C_MODEL, self.failure_input = self.get_plant()
        self.FDD_switch_time = 60
        self.failure_time = 10
        self.task = task()
        self.task_fun, self.evaluation, self.FDD = self.task.choose_task(evaluation, self.failure_input, FDD)

        self.time = self.task_fun()[3]
        self.dt = self.time[1] - self.time[0]
        self.ref_signal = self.task_fun()[0]
        self.track_indices = self.task_fun()[1]
        self.obs_indices = self.task_fun()[2]

        self.sideslip_factor, self.pitch_factor, self.roll_factor, self.alt_factor = self.adapt_to_failure()

        self.observation_space = gym.spaces.Box(-100, 100, shape=(len(self.obs_indices) + 3,), dtype=np.float64)
        self.action_space = gym.spaces.Box(-1., 1., shape=(3,), dtype=np.float64)
        self.current_deflection = np.zeros(3)

        self.agents, self.agentID = self.load_agent(FDD)
        # self.agents, self.agentID = None, None

        self.state = None
        self.state_deg = None
        self.scale_s = None
        self.state_history = None
        self.action_history = None
        # self.action_history_filtered = None
        self.error = None
        self.step_count = None
        self.external_ref_signal = None

    def step(self, action_rates: np.ndarray):

        # self.current_deflection = self.bound_a(self.current_deflection + self.scale_a(action_rates) * self.dt) #diff: bound_a
        self.current_deflection = self.current_deflection + self.scale_a(action_rates) * self.dt  # diff: bound_a

        if self.sideslip_factor[self.step_count - 1] == 0.0: self.current_deflection[2] = 0.0

        if self.time[self.step_count] < self.failure_time and self.evaluation:
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
            self.error[self.track_indices.index(9)] *= self.alt_factor[self.step_count]

        self.state_history[:, self.step_count] = self.state_deg
        self.action_history[:, self.step_count] = self.current_deflection
        # self.action_history_filtered[:, self.step_count] = filtered_deflection

        self.step_count += 1
        done = bool(self.step_count >= self.time.shape[0])
        if np.isnan(self.state).sum() > 0:
            if not self.evaluation:
                ID = get_ID(6)
                agent = SAC.load("agent/trained/tmp/best_model.zip", env=self)
                agent.ID = ID
                agent.save(f'agent/trained/{self.task_fun()[4]}_{agent.ID}.zip')
            # print(self.state_history[:, self.step_count - 2], self.time[self.step_count - 1])
            plot_response('before_crash', self, self.task_fun(), 100, during_training=False,
                          failure=self.failure_input[0], FDD=self.FDD, broken=True)
            exit()
        # if self.state[9] <= 20.0 or self.state[9] >= 1e4 or np.greater(np.abs(r2d(self.state[:3])), 1e4).any() \
        #         or np.greater(np.abs(r2d(self.state[6:9])), 1e3).any():
        #     print('Encountered crash. Episode terminated early.')
        #     return np.zeros(self.observation_space.shape), -1, True, {'is_success': False}

        return self.get_obs(), self.get_reward(), done, {'is_success': True}

    def reset(self):

        self.reset_soft()
        self.ref_signal = self.task_fun()[0]
        return np.zeros(self.observation_space.shape)

    def reset_soft(self):

        self.C_MODEL.initialize()
        action_trim = np.array(
            [0, 0, 0, 0., 0., 0., 0., 0.,
             0, 0, self.failure_input[1]])
        # action_trim = np.array(
        #     [-0.024761262011031245, 1.3745996716698875e-14, -7.371050575286063e-14, 0., 0., 0., 0., 0.,
        #      0.38576210972746433, 0.38576210972746433, self.failure_input[1]])
        self.state = self.C_MODEL.step(action_trim)
        self.scale_s = np.ones(self.state.shape)
        self.scale_s[[0, 1, 2, 4, 5, 6, 7, 8]] = 180 / np.pi
        self.state_deg = self.state * self.scale_s
        self.state_history = np.zeros((self.state.shape[0], self.time.shape[0]))
        self.action_history = np.zeros((self.action_space.shape[0], self.time.shape[0]))
        # self.action_history_filtered = self.action_history.copy()
        self.error = np.zeros(len(self.track_indices))
        self.step_count = 0
        self.current_deflection = np.zeros(3)
        return np.zeros(self.observation_space.shape)

    def get_reward(self):

        max_bound = np.ones(self.error.shape)
        reward_vec = np.abs(np.maximum(np.minimum(r2d(self.error / 30)**2, max_bound), -max_bound))
        # reward_vec = np.abs(np.maximum(np.minimum(r2d(self.error / 30), max_bound), -max_bound))
        # reward_vec = -np.maximum(np.minimum(1 / (np.abs(self.error) * 10 + 1), max_bound), -max_bound)
        # reward_vec = -1 / (np.abs(self.error) * 10 + 1)
        # reward_vec = np.abs(r2d(self.error / 30))
        # reward_vec = r2d(self.error) ** 2

        reward = -reward_vec.sum() / self.error.shape[0]
        return reward

    def get_obs(self):

        untracked_obs_index = np.setdiff1d(self.obs_indices, self.track_indices)
        return np.hstack([self.error, self.state[untracked_obs_index], self.current_deflection])

    def get_RMSE(self):

        assert bool(self.step_count >= self.time.shape[0]), \
            f'Error: cannot obtain RMSE before episode is completed. Current time is {self.time[self.step_count]}s.'
        y_ref = self.ref_signal.copy()
        y_ref2 = self.ref_signal.copy()
        y_meas = self.state_history[self.track_indices, :].copy()
        y_ref2[-1, 0] = 5
        y_ref2[-1, 1] = -5

        RMSE = np.sqrt(np.mean(np.square((y_ref - y_meas)), axis=1))/(y_ref2.max(axis=1)-y_ref2.min(axis=1))
        return RMSE

    def get_MAE(self):

        assert bool(self.step_count >= self.time.shape[0]), \
            f'Error: cannot obtain MAE before episode is completed. Current time is {self.time[self.step_count]}s.'
        y_ref = self.ref_signal.copy()
        y_ref2 = self.ref_signal.copy()
        y_meas = self.state_history[self.track_indices, :].copy()
        y_ref2[-1, 0] = 5
        y_ref2[-1, 1] = -5

        MAE = np.mean(np.absolute(y_ref - y_meas), axis=1)/(y_ref2.max(axis=1)-y_ref2.min(axis=1))
        return MAE

    def scale_a(self, action_unscaled: np.ndarray) -> np.ndarray:
        """Min-max un-normalization from [-1, 1] action space to actuator limits"""

        return unscale_action(self.rate_limits, action_unscaled)

    def bound_a(self, action):

        return np.minimum(np.maximum(action, self.deflection_limits.low), self.deflection_limits.high)

    @abstractmethod
    def get_plant(self):
        pass

    @abstractmethod
    def load_agent(self, FDD):
        pass

    def adapt_to_failure(self):

        pitch_factor = np.ones(self.time.shape[0])
        roll_factor = np.ones(self.time.shape[0])
        alt_factor = 0.25 * np.ones(self.time.shape[0])
        if self.evaluation:
            sideslip_factor = 4.0 * np.ones(self.time.shape[0])
            if self.task_fun()[4] == 'altitude_2attitude':
                roll_factor = 2 * np.ones(self.time.shape[0])
        else:
            sideslip_factor = 10.0 * np.ones(self.time.shape[0])

        return sideslip_factor, pitch_factor, roll_factor, alt_factor

    def FFD_change(self):
        pass

    def render(self, ext_agent=None, verbose=1):

        during_training = False
        if ext_agent is not None:
            self.agents = [ext_agent]
            # self.agents.save(f'agent/trained/{self.task_fun()[4]}_last.zip')
            self.agentID = 'last'
            verbose = 0
            during_training = True

        if self.FDD:
            self.reset()
            agent_robust = self.agents[0]
            agent_adaptive = self.agents[1]
        else:
            agent_robust = self.agents[0]
            agent_adaptive = None

        obs = self.reset_soft()
        return_a = 0
        done = False
        items = range(self.time.shape[0])
        with alive_bar(len(items)) as bar:
            while not done:
                if self.time[self.step_count] < self.FDD_switch_time or not self.FDD:
                    action, _ = agent_robust.predict(obs, deterministic=True)
                else:
                    self.FFD_change()
                    action, _ = agent_adaptive.predict(obs, deterministic=True)
                obs, reward, done, info = self.step(action)
                return_a += reward
                bar()

        plot_response(self.agentID, self, self.task_fun(), return_a, during_training,
                      self.failure_input[0], FDD=self.FDD)
        if verbose > 0:
            print(f'Goal reached! Return = {return_a:.2f}')
            np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
            print(f'nRMSE% avg: {(self.get_RMSE().sum()) / 3 * 100:.2f}%')
            print(f'nMAE% avg: {(self.get_MAE().sum()) / 3 * 100:.2f}%')
            print('')

    def close(self):
        self.C_MODEL.terminate()
        return

    class ActionLimits:

        def __init__(self, limits):
            self.low, self.high = limits[0, :], limits[1, :]


class CitationNormal(Citation):

    def get_plant(self):
        plant = importlib.import_module(f'envs.normal_2000_90._citation', package=None)
        return plant, ['normal', 1.0, 1.0]

    def load_agent(self, FDD=False):
        if FDD:
            raise NotImplementedError('No fault detection and diagnosis on the non-failed system.')
        return [SAC.load(f"agent/trained/{self.task.agent_catalog['normal']}.zip",
                         env=self)], self.task.agent_catalog['normal']


class CitationRudderStuck(Citation):

    def get_plant(self):

        plant = importlib.import_module(f'envs.dr._citation', package=None)
        return plant, ['dr', 0.0, -15.0]

    def load_agent(self, FDD):
        if FDD:
            return [CitationNormal().load_agent()[0][0],
                    SAC.load(f"agent/trained/{self.task.agent_catalog['rudder_stuck']}.zip", env=self)], \
                   self.task.agent_catalog['rudder_stuck']
        return CitationNormal().load_agent()
        # return SAC.load(f"agent/trained/{self.task.agent_catalog['rudder_stuck']}.zip", env=self),
        # self.task.agent_catalog['rudder_stuck']

    def adapt_to_failure(self):

        sideslip_factor, pitch_factor, roll_factor, alt_factor = super(CitationRudderStuck, self).adapt_to_failure()
        if self.FDD:
            sideslip_factor[np.argwhere(self.time == self.FDD_switch_time)[0, 0]:] *= 0.0
            roll_factor[np.argwhere(self.time == self.FDD_switch_time)[0, 0]:] *= 0.5

        return sideslip_factor, pitch_factor, roll_factor, alt_factor


class CitationAileronEff(Citation):

    def get_plant(self):

        plant = importlib.import_module(f'envs.da._citation', package=None)
        return plant, ['da', 1.0, 0.3]

    def load_agent(self, FDD):
        if FDD:
            return [CitationNormal().load_agent()[0][0],
                    SAC.load(f"agent/trained/{self.task.agent_catalog['aileron_eff']}.zip", env=self)], \
                   self.task.agent_catalog['aileron_eff']
        return CitationNormal().load_agent()
        # return SAC.load(f"agent/trained/{self.task.agent_catalog['rudder_stuck']}.zip", env=self),
        # self.task.agent_catalog['rudder_stuck']

    def adapt_to_failure(self):

        sideslip_factor, pitch_factor, roll_factor, alt_factor = super(CitationAileronEff, self).adapt_to_failure()
        if self.FDD:
            pitch_factor[np.argwhere(self.time == self.FDD_switch_time)[0, 0]:] *= 1.5

        return sideslip_factor, pitch_factor, roll_factor, alt_factor


class CitationElevRange(Citation):

    def get_plant(self):
        plant = importlib.import_module(f'envs.de._citation', package=None)
        # self.deflection_limits = self.ActionLimits(np.array([[-3.0, -37.24, -21.77], [3.0, 37.24, 21.77]]))
        # self.rate_limits = self.ActionLimits(np.array([[-7, -40, -20], [7, 40, 20]]))
        return plant, ['de', 20.05, 3.0]

    def load_agent(self, FDD):
        if FDD:
            return [CitationNormal().load_agent()[0][0],
                    SAC.load(f"agent/trained/{self.task.agent_catalog['elev_range']}.zip", env=self)], \
                   self.task.agent_catalog['elev_range']
        return CitationNormal().load_agent()
        # return SAC.load(f"agent/trained/{self.task.agent_catalog['elev_range']}.zip", env=self),
        # self.task.agent_catalog['elev_range']

    def FFD_change(self):
        self.deflection_limits = self.ActionLimits(np.array([[-3.0, -37.24, -21.77], [3.0, 37.24, 21.77]]))
        self.rate_limits = self.ActionLimits(np.array([[-7, -40, -20], [7, 40, 20]]))


class CitationCgShift(Citation):

    def get_plant(self):

        plant = importlib.import_module(f'envs.cg._citation', package=None)
        return plant, ['cg', 1.0, 1.04]

    def load_agent(self, FDD):
        if FDD:
            return [CitationNormal().load_agent()[0][0],
                    SAC.load(f"agent/trained/{self.task.agent_catalog['cg_shift']}.zip", env=self)], \
                   self.task.agent_catalog['cg_shift']
        return CitationNormal().load_agent()
        # return SAC.load(f"agent/trained/{self.task.agent_catalog['cg_shift']}.zip", env=self),
        # self.task.agent_catalog[ 'cg_shift']

    def adapt_to_failure(self):
        sideslip_factor, pitch_factor, roll_factor, alt_factor = super(CitationCgShift, self).adapt_to_failure()
        if self.FDD:
            alt_factor[np.argwhere(self.time == self.FDD_switch_time)[0, 0]:] *= 0.5

        return sideslip_factor, pitch_factor, roll_factor, alt_factor


class CitationIcing(Citation):

    def get_plant(self):

        plant = importlib.import_module(f'envs.ice2000._citation', package=None)
        return plant, ['ice', 1.0, 0.7]  # https://doi.org/10.1016/S0376-0421(01)00018-5

    def load_agent(self, FDD):
        if FDD:
            return [CitationNormal().load_agent()[0][0],
                    SAC.load(f"agent/trained/{self.task.agent_catalog['icing']}.zip", env=self)], \
                   self.task.agent_catalog['icing']
        return CitationNormal().load_agent()

    # return SAC.load(f"agent/trained/{self.task.agent_catalog['icing']}.zip", env=self), self.task.agent_catalog['icing']

    def reset(self):
        super(CitationIcing, self).reset()
        self.ref_signal = self.task_fun(theta_angle=25)[0]
        # if 9 in self.track_indices:
        #     self.ref_signal[0, :] += 2000.0

        return np.zeros(self.observation_space.shape)

    def adapt_to_failure(self):

        sideslip_factor, pitch_factor, roll_factor, alt_factor = super(CitationIcing, self).adapt_to_failure()

        if self.FDD:
            alt_factor[np.argwhere(self.time == self.FDD_switch_time)[0, 0]:] *= 0.25

        return sideslip_factor, pitch_factor, roll_factor, alt_factor


class CitationHorzTail(Citation):

    def get_plant(self):

        plant = importlib.import_module(f'envs.ht._citation', package=None)
        return plant, ['ht', 1.0, 0.3]

    def load_agent(self, FDD):
        if FDD:
            return [CitationNormal().load_agent()[0][0],
                    SAC.load(f"agent/trained/{self.task.agent_catalog['horz_tail']}.zip", env=self)], \
                   self.task.agent_catalog['horz_tail']
        return CitationNormal().load_agent()
        # return [SAC.load(f"agent/trained/{self.task.agent_catalog['horz_tail']}.zip", env=self)], \
        #        self.task.agent_catalog['horz_tail']

    def adapt_to_failure(self):

        sideslip_factor, pitch_factor, roll_factor, alt_factor = super(CitationHorzTail, self).adapt_to_failure()
        if self.FDD:
            alt_factor[np.argwhere(self.time == self.FDD_switch_time)[0, 0]:] *= 0.01

        return sideslip_factor, pitch_factor, roll_factor, alt_factor


class CitationVertTail(Citation):

    def get_plant(self):

        plant = importlib.import_module(f'envs.vt._citation', package=None)
        return plant, ['vt', 1.0, 0.3]

    def load_agent(self, FDD):
        if FDD:
            return [CitationNormal().load_agent()[0][0],
                    SAC.load(f"agent/trained/{self.task.agent_catalog['normal']}.zip", env=self)], \
                   self.task.agent_catalog['vert_tail']
        return CitationNormal().load_agent()
        # return [SAC.load(f"agent/trained/{self.task.agent_catalog['vert_tail']}.zip", env=self)],\
        #        self.task.agent_catalog['vert_tail']

    def adapt_to_failure(self):

        sideslip_factor, pitch_factor, roll_factor, alt_factor = super(CitationVertTail, self).adapt_to_failure()
        if self.FDD:
            sideslip_factor[np.argwhere(self.time == self.FDD_switch_time)[0, 0]:] *= 0.25

        return sideslip_factor, pitch_factor, roll_factor, alt_factor


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
