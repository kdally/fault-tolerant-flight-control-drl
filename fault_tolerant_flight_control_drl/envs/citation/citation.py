import gym
import numpy as np
from abc import abstractmethod

from fault_tolerant_flight_control_drl.agent import SAC
from fault_tolerant_flight_control_drl.tools import AltitudeTask, AttitudeTask, BodyRateTask
from fault_tolerant_flight_control_drl.tools import ReliabilityTask, DisturbanceRejectionAtt
from fault_tolerant_flight_control_drl.tools import plot_response
import importlib
from fault_tolerant_flight_control_drl.tools.math_util import unscale_action, d2r, r2d
from fault_tolerant_flight_control_drl.tools import get_ID
from alive_progress import alive_bar


class Citation(gym.Env):
    """
    Citation environment that follows the gym.env interface
    Developed to be interfaced with a modified version of the CitAST environment, built with the DASMAT model and owned
    by the Delft University of Technology. Follow the 'CitAST for Python' instructions at
    https://github.com/kdally/fault-tolerant-flight-control-drl/blob/master/docs/CitAST_for_Python.pdf for installation.
    Author: Killian Dally

    :param evaluation: (bool) If False, the environment will be given training-specific shorter tasks.
        If True, the environment is given longer and unseen tasks as part of the evaluation.
    :param FDD: (bool) If True, the Fault Detection and Diagnosis module is added which switches from robust to
        adaptive control at self.FDD_switch_time.
    :param task: (Task) one of AltitudeTask, AttitudeTask, BodyRateTask, ReliabilityTask, DisturbanceRejection
    :param disturbance: (bool) If True, disturbance forces are added in the environment. Normal disturbance values from
    https://doi.org/10.2514/6.2018-1127.
    :param sensor_noise: (bool) If True, sensor noise is added to the environment observations based on the sensor noise
        estimates of the Cessna Citation 550 given in https://doi.org/10.2514/6.2018-1127.
    :param low_pass: (bool) It True, control inputs are filtered with a first-order low-pass filter.
    :param init_alt: (float) Initial flight altitude. One of 2000 or 5000.
    :param init_speed: (float) Initial speed. One of 90 or 140.
    """

    def __init__(self, evaluation=False, FDD=False, task=AttitudeTask,
                 disturbance=False, sensor_noise=False, low_pass=False,
                 init_alt=2000, init_speed=90):
        super(Citation, self).__init__()

        assert bool((FDD and init_alt == 2000 and init_speed == 90) or not FDD), \
            'Failure cases only implemented for initial conditions init_alt == 2000 & init_speed == 90'

        self.rate_limits = self.ActionLimits(np.array([[-20, -40, -20], [20, 40, 20]]))
        self.deflection_limits = self.ActionLimits(np.array([[-20.05, -37.24, -21.77], [14.9, 37.24, 21.77]]))
        self.placeholder_cond = False
        self.C_MODEL, self.failure_input = self.get_plant()
        self.FDD_switch_time = 60
        self.failure_time = 10
        self.task = task()
        self.task_fun, self.evaluation, self.FDD = self.task.choose_task(evaluation, self.failure_input, FDD)
        self.has_sensor_noise = sensor_noise
        self.has_disturbance = disturbance
        self.enable_low_pass = low_pass

        self.time = self.task_fun()[3]
        self.dt = self.time[1] - self.time[0]
        self.ref_signal = self.task_fun(init_alt=init_alt)[0]
        self.track_indices = self.task_fun()[1]
        self.obs_indices = self.task_fun()[2]

        self.sideslip_factor, self.pitch_factor, self.roll_factor, self.alt_factor = self.adapt_to_failure()

        self.observation_space = gym.spaces.Box(-100, 100, shape=(len(self.obs_indices) + 3,), dtype=np.float64)
        self.action_space = gym.spaces.Box(-1., 1., shape=(3,), dtype=np.float64)
        self.current_deflection = np.zeros(3)

        self.agent_path = 'fault_tolerant_flight_control_drl/agent/trained'
        self.agents, self.agentID = self.load_agent(FDD)  # type: SAC
        # self.agents, self.agentID = None, None

        self.state = None
        self.state_deg = None
        self.scale_s = None
        self.state_history = None
        self.action_history = None
        self.error = None
        self.step_count = None
        self.external_ref_signal = None

    def step(self, action_rates: np.ndarray):

        self.current_deflection = self.current_deflection + self.scale_a(action_rates) * self.dt
        if self.sideslip_factor[self.step_count - 1] == 0.0: self.current_deflection[2] = 0.0
        filtered_deflection = self.filter_control_input(self.current_deflection)

        if self.time[self.step_count] < self.failure_time and self.evaluation:
            self.state = self.C_MODEL.step(
                np.hstack([d2r(filtered_deflection + self.add_disturbance()[:, self.step_count]), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                           self.failure_input[1]]))
        else:
            self.state = self.C_MODEL.step(
                np.hstack([d2r(filtered_deflection + self.add_disturbance()[:, self.step_count]), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                           self.failure_input[2]]))
        self.state_deg = self.state * self.scale_s

        self.error = d2r(self.ref_signal[:, self.step_count] -
                         self.state_deg[self.track_indices] + self.get_sensor_noise()[self.track_indices]) \
                       * self.scale_error(self.step_count)

        self.state_history[:, self.step_count] = self.state_deg
        self.action_history[:, self.step_count] = filtered_deflection

        self.step_count += 1
        done = bool(self.step_count >= self.time.shape[0])

        if np.isnan(self.state).sum() > 0:
            self.stop_NaNs()

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
        # reward_vec = np.abs(np.maximum(np.minimum(r2d(self.error / 30)**2, max_bound), -max_bound))  # square function
        reward_vec = np.abs(np.maximum(np.minimum(r2d(self.error / 30), max_bound), -max_bound))  # rational function
        # reward_vec = - np.maximum(np.minimum(1 / (np.abs(self.error) * 10 + 1), max_bound),
        #                          - max_bound)  # abs. linear function
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

        RMSE = np.sqrt(np.mean(np.square((y_ref - y_meas)), axis=1)) / (y_ref2.max(axis=1) - y_ref2.min(axis=1))
        return RMSE

    def get_MAE(self):

        assert bool(self.step_count >= self.time.shape[0]), \
            f'Error: cannot obtain MAE before episode is completed. Current time is {self.time[self.step_count]}s.'
        y_ref = self.ref_signal.copy()
        y_ref2 = self.ref_signal.copy()
        y_meas = self.state_history[self.track_indices, :].copy()
        y_ref2[-1, 0] = 5
        y_ref2[-1, 1] = -5

        MAE = np.mean(np.absolute(y_ref - y_meas), axis=1) / (y_ref2.max(axis=1) - y_ref2.min(axis=1))
        return MAE

    def stop_NaNs(self):
        print('Encountered crash. Episode terminated early.')
        if not self.evaluation:
            ID = get_ID(6)
            agent = SAC.load("fault_tolerant_flight_control_drl/agent/trained/tmp/best_model.zip", env=self)
            agent.ID = ID
            agent.save(f'{self.agent_path}/{self.task_fun()[4]}_{agent.ID}.zip')
            print('Training is corrupt because of NaN values, terminated early. '
                  'So-far best trained agent may show good performance.')
        plot_response('before_crash', self, self.task_fun(), 100, during_training=False,
                      failure=self.failure_input[0], FDD=self.FDD, broken=True)
        exit()

    def filter_control_input(self, deflection):

        w_0 = 2 * 2 * np.pi  # rad/s
        filtered_deflection = deflection.copy()
        if self.step_count > 1 and self.enable_low_pass:
            filtered_deflection = self.action_history[:, self.step_count - 1] / (1 + w_0 * self.dt) + \
                                  deflection * (w_0 * self.dt) / (1 + w_0 * self.dt)

        return filtered_deflection

    def get_sensor_noise(self):

        # values in degrees, SSD
        sensor_noise = np.zeros(self.state.shape)
        if self.has_sensor_noise:

            # p, q, r measurement from https://doi.org/10.2514/6.2018-0385
            sensor_noise[0:3] += r2d(np.random.normal(scale=np.sqrt(4.0e-7), size=3)+3.0e-5)

            # sideslip, estimate from https://doi.org/10.2514/6.2018-0385
            sensor_noise[5] += r2d(np.random.normal(scale=np.sqrt(7.5e-8))+1.8e-3)

            # phi, theta measurement from https://doi.org/10.2514/6.2018-0385
            sensor_noise[6:8] += r2d(np.random.normal(scale=np.sqrt(1e-9), size=2)+4.0e-3)

            # h estimate from https://doi.org/10.2514/6.2018-0385
            sensor_noise[9] += np.random.normal(scale=np.sqrt(4.5e-3))+8.0e-3
        return sensor_noise

    def add_disturbance(self):

        disturbance = np.zeros((self.action_space.shape[0], self.time.shape[0]))
        if self.has_disturbance:  # 3211 input in deg
            disturbance[0, np.argwhere(self.time == 1)[0, 0]:np.argwhere(self.time == 4)[0, 0]] = 0.5
            disturbance[0, np.argwhere(self.time == 4)[0, 0]:np.argwhere(self.time == 6)[0, 0]] = -0.9
            disturbance[0, np.argwhere(self.time == 6)[0, 0]:np.argwhere(self.time == 7)[0, 0]] = 1.2
            disturbance[0, np.argwhere(self.time == 7)[0, 0]:np.argwhere(self.time == 8)[0, 0]] = -1.2

            disturbance[1, np.argwhere(self.time == 10)[0, 0]:np.argwhere(self.time == 13)[0, 0]] = -0.5
            disturbance[1, np.argwhere(self.time == 13)[0, 0]:np.argwhere(self.time == 15)[0, 0]] = 0.9
            disturbance[1, np.argwhere(self.time == 15)[0, 0]:np.argwhere(self.time == 16)[0, 0]] = -1.2
            disturbance[1, np.argwhere(self.time == 16)[0, 0]:np.argwhere(self.time == 17)[0, 0]] = 1.2

        return disturbance

    def scale_error(self, step_count):

        if 7 in self.track_indices:
            return np.array([self.pitch_factor[step_count],
                             self.roll_factor[step_count], self.sideslip_factor[step_count]])
        else:
            return np.array([self.alt_factor[step_count],
                             self.roll_factor[step_count], self.sideslip_factor[step_count]])

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
        alt_factor = np.ones(self.time.shape[0])
        if self.evaluation:
            sideslip_factor = 4.0 * np.ones(self.time.shape[0])
            if self.task_fun()[4] == 'altitude_2attitude':
                roll_factor *= 2
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
            # print(f'Goal reached! Return = {return_a:.2f}')
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
    """
    Normal Citation Dynamics class, a sub-class of the Citation class.
    Author: Killian Dally

    :param evaluation: (bool) If False, the environment will be given training-specific shorter tasks.
    If True, the environment is given longer and unseen tasks as part of the evaluation.
    :param FDD: (bool) If True, the Fault Detection and Diagnosis module is added which switches from robust to
    adaptive control at self.FDD_switch_time.
    :param task: (Task) one of AltitudeTask, AttitudeTask, BodyRateTask, ReliabilityTask
    :param disturbance: (bool) If True, disturbance forces are added in the environment. Normal disturbance values from
    https://doi.org/10.2514/6.2018-1127.
    :param sensor_noise: (bool) If True, sensor noise is added to the environment observations based on the sensor noise
    estimates of the Cessna Citation 550 given in https://doi.org/10.2514/6.2018-1127.
    :param low_pass: (bool) It True, control inputs are filtered with a first-order low-pass filter.
    :param init_alt: (float) Initial flight altitude. One of 2000 or 5000.
    :param init_speed: (float) Initial speed. One of 90 or 140.
    """

    def __init__(self, init_alt=2000, init_speed=90, evaluation=False, FDD=False, task=AttitudeTask,
                 disturbance=False, sensor_noise=False, low_pass=False):
        self.init_alt = init_alt
        self.init_speed = init_speed
        super(CitationNormal, self).__init__(evaluation=evaluation, FDD=FDD, task=task,
                                             disturbance=disturbance, sensor_noise=sensor_noise, low_pass=low_pass)
        self.ref_signal = self.task_fun(init_alt=init_alt)[0]

    def get_plant(self):

        path = 'fault_tolerant_flight_control_drl.envs.citation'
        if self.init_alt == 2000 and self.init_speed == 90:
            plant = importlib.import_module(f'{path}.normal_2000_90._citation', package=None)
        elif self.init_alt == 2000 and self.init_speed == 140:
            plant = importlib.import_module(f'{path}.normal_2000_140._citation', package=None)
            self.placeholder_cond = True
        elif self.init_alt == 5000 and self.init_speed == 90:
            plant = importlib.import_module(f'{path}.normal_5000_90._citation', package=None)
        elif self.init_alt == 5000 and self.init_speed == 140:
            plant = importlib.import_module(f'{path}.normal_5000_140._citation', package=None)
        else:
            raise NotImplementedError('No model with the specified initial conditions is present. ' \
                                      'Choose within init_alt={2000, 5000} and init_speed={90, 120}.')

        return plant, ['normal', 1.0, 1.0]

    def load_agent(self, FDD=False):
        if FDD:
            raise NotImplementedError('No fault detection and diagnosis on the non-failed system.')
        return [SAC.load(f"{self.agent_path}/{self.task.agent_catalog['normal']}.zip",
                         env=self)], self.task.agent_catalog['normal']

    def reset(self):
        super(CitationNormal, self).reset()
        self.ref_signal = self.task_fun(init_alt=self.init_alt)[0]
        return np.zeros(self.observation_space.shape)

    def reset_soft(self):
        super(CitationNormal, self).reset_soft()
        self.ref_signal = self.task_fun(init_alt=self.init_alt)[0]
        return np.zeros(self.observation_space.shape)


class CitationRudderStuck(Citation):
    """
    Citation Dynamics class with rudder failure, a sub-class of the Citation class.
    The rudder is stuck at -15deg starting from self.failure_time.
    Author: Killian Dally
    """

    def get_plant(self):
        plant = importlib.import_module(f'fault_tolerant_flight_control_drl.envs.citation.dr._citation', package=None)
        return plant, ['dr', 0.0, -15.0]

    def load_agent(self, FDD):
        if FDD:
            return [SAC.load(f"{self.agent_path}/{self.task.agent_catalog['normal']}.zip", env=self),
                    SAC.load(f"{self.agent_path}/{self.task.agent_catalog['rudder_stuck']}.zip", env=self)], \
                   self.task.agent_catalog['rudder_stuck']
        return CitationNormal().load_agent()

    def adapt_to_failure(self):

        sideslip_factor, pitch_factor, roll_factor, alt_factor = super(CitationRudderStuck, self).adapt_to_failure()
        if self.FDD:
            sideslip_factor[np.argwhere(self.time == self.FDD_switch_time)[0, 0]:] *= 0.0
            roll_factor[np.argwhere(self.time == self.FDD_switch_time)[0, 0]:] *= 0.5

        return sideslip_factor, pitch_factor, roll_factor, alt_factor


class CitationAileronEff(Citation):
    """
    Citation Dynamics class with aileron failure, a sub-class of the Citation class.
    The aileron effectiveness is reduced by 70% from self.failure_time.
    Author: Killian Dally
    """

    def get_plant(self):

        plant = importlib.import_module(f'fault_tolerant_flight_control_drl.envs.citation.da._citation', package=None)
        return plant, ['da', 1.0, 0.3]

    def load_agent(self, FDD):
        if FDD:
            return [SAC.load(f"{self.agent_path}/{self.task.agent_catalog['normal']}.zip", env=self),
                    SAC.load(f"{self.agent_path}/{self.task.agent_catalog['aileron_eff']}.zip", env=self)], \
                   self.task.agent_catalog['aileron_eff']
        return CitationNormal().load_agent()

    def adapt_to_failure(self):

        sideslip_factor, pitch_factor, roll_factor, alt_factor = super(CitationAileronEff, self).adapt_to_failure()
        if self.FDD:
            pitch_factor[np.argwhere(self.time == self.FDD_switch_time)[0, 0]:] *= 1.5

        return sideslip_factor, pitch_factor, roll_factor, alt_factor


class CitationElevRange(Citation):
    """
    Citation Dynamics class with elevator failure, a sub-class of the Citation class.
    The elevator operating range is reduced to [-3 deg, 3 deg] from self.failure_time.
    Author: Killian Dally
    """

    def get_plant(self):
        plant = importlib.import_module(f'fault_tolerant_flight_control_drl.envs.citation.de._citation', package=None)
        return plant, ['de', 20.05, 2.5]

    def load_agent(self, FDD):
        if FDD:
            return [SAC.load(f"{self.agent_path}/{self.task.agent_catalog['normal']}.zip", env=self),
                    SAC.load(f"{self.agent_path}/{self.task.agent_catalog['elev_range']}.zip", env=self)], \
                   self.task.agent_catalog['elev_range']
        return CitationNormal().load_agent()

    def FFD_change(self):
        self.deflection_limits = self.ActionLimits(np.array([[-3.0, -37.24, -21.77], [3.0, 37.24, 21.77]]))
        self.rate_limits = self.ActionLimits(np.array([[-7, -40, -20], [7, 40, 20]]))


class CitationCgShift(Citation):
    """
    Citation Dynamics class with backwards c.g. shift, a sub-class of the Citation class.
    A 300kg payload moving from the from the front to the back of the passenger cabin is simulated,
    which translates to a backwards c.g. shift of 0.25m from self.failure_time.
    Author: Killian Dally
    """

    def get_plant(self):

        plant = importlib.import_module(f'fault_tolerant_flight_control_drl.envs.citation.cg._citation', package=None)
        return plant, ['cg', 1.0, 1.04]

    def load_agent(self, FDD):
        if FDD:
            return [SAC.load(f"{self.agent_path}/{self.task.agent_catalog['normal']}.zip", env=self),
                    SAC.load(f"{self.agent_path}/{self.task.agent_catalog['cg_shift']}.zip", env=self)], \
                   self.task.agent_catalog['cg_shift']
        return CitationNormal().load_agent()

    def adapt_to_failure(self):
        sideslip_factor, pitch_factor, roll_factor, alt_factor = super(CitationCgShift, self).adapt_to_failure()
        if self.FDD:
            alt_factor[np.argwhere(self.time == self.FDD_switch_time)[0, 0]:] *= 0.5

        return sideslip_factor, pitch_factor, roll_factor, alt_factor


class CitationIcing(Citation):
    """
   Citation Dynamics class with icing, a sub-class of the Citation class.
   A large accumulation of ice on the wing is simulated according to the measurements made
   in https://doi.org/10.1016/S0376-0421(01)00018-5 from self.failure_time. In practice, C_L_max and alpha_stall are
   reduced by 30% and C_D increased by 0.06.
   Author: Killian Dally
   """

    def get_plant(self):

        plant = importlib.import_module(f'fault_tolerant_flight_control_drl.envs.citation.ice._citation', package=None)
        return plant, ['ice', 1.0, 0.7]  # https://doi.org/10.1016/S0376-0421(01)00018-5

    def load_agent(self, FDD):
        if FDD:
            return [SAC.load(f"{self.agent_path}/{self.task.agent_catalog['normal']}.zip", env=self),
                    SAC.load(f"{self.agent_path}/{self.task.agent_catalog['icing']}.zip", env=self,
                             policy_kwargs=dict(layers=[32, 32]))], \
                   self.task.agent_catalog['icing']
        return CitationNormal().load_agent()

    def reset(self):
        super(CitationIcing, self).reset()
        self.ref_signal = self.task_fun()[0]

        return np.zeros(self.observation_space.shape)

    def adapt_to_failure(self):

        sideslip_factor, pitch_factor, roll_factor, alt_factor = super(CitationIcing, self).adapt_to_failure()

        if self.FDD:
            alt_factor[np.argwhere(self.time == self.FDD_switch_time)[0, 0]:] *= 0.25

        return sideslip_factor, pitch_factor, roll_factor, alt_factor


class CitationHorzTail(Citation):
    """
   Citation Dynamics class with partial horizontal tail loss, a sub-class of the Citation class.
   Author: Killian Dally
   """

    def get_plant(self):

        plant = importlib.import_module(f'fault_tolerant_flight_control_drl.envs.citation.ht._citation', package=None)
        return plant, ['ht', 1.0, 0.3]

    def load_agent(self, FDD):
        if FDD:
            return [SAC.load(f"{self.agent_path}/{self.task.agent_catalog['normal']}.zip", env=self),
                    SAC.load(f"{self.agent_path}/{self.task.agent_catalog['horz_tail']}.zip", env=self,
                             policy_kwargs=dict(layers=[32, 32]))], \
                   self.task.agent_catalog['horz_tail']
        return CitationNormal().load_agent()

    def adapt_to_failure(self):

        sideslip_factor, pitch_factor, roll_factor, alt_factor = super(CitationHorzTail, self).adapt_to_failure()
        if self.FDD:
            alt_factor[np.argwhere(self.time == self.FDD_switch_time)[0, 0]:] *= 0.01

        return sideslip_factor, pitch_factor, roll_factor, alt_factor


class CitationVertTail(Citation):
    """
   Citation Dynamics class with partial vertical tail loss, a sub-class of the Citation class.
   Author: Killian Dally
   """

    def get_plant(self):

        plant = importlib.import_module(f'fault_tolerant_flight_control_drl.envs.citation.vt._citation', package=None)
        return plant, ['vt', 1.0, 0.3]

    def load_agent(self, FDD):
        if FDD:
            return [SAC.load(f"{self.agent_path}/{self.task.agent_catalog['normal']}.zip", env=self),
                    SAC.load(f"{self.agent_path}/{self.task.agent_catalog['normal']}.zip", env=self)], \
                   self.task.agent_catalog['vert_tail']
        return CitationNormal().load_agent()

    def adapt_to_failure(self):

        sideslip_factor, pitch_factor, roll_factor, alt_factor = super(CitationVertTail, self).adapt_to_failure()
        if self.FDD:
            sideslip_factor[np.argwhere(self.time == self.FDD_switch_time)[0, 0]:] *= 0.25

        return sideslip_factor, pitch_factor, roll_factor, alt_factor


class CitationDistAlpha(CitationNormal):
    """
       CitationNormal Dynamics class with atmospheric disturbances as verital .
       The rudder is stuck at -15deg starting from self.failure_time.
       Author: Killian Dally
    """
    def get_plant(self):

        path = 'fault_tolerant_flight_control_drl.envs.citation'
        if self.init_alt == 2000 and self.init_speed == 90:
            plant = importlib.import_module(f'{path}.normal_2000_90_dist._citation', package=None)
        else:
            raise NotImplementedError('No model with the specified initial conditions is present.')
        return plant, ['normal', 1.0, 1.0]


class CitationVerif(CitationNormal):
    """
   Normal Citation Dynamics class for verification, a sub-class of the Citation class.
   It emulates MATLAB from Python to compare the response of the compiled model and that of the Simulink model.
   Author: Killian Dally
   """

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

        return self.get_obs(), self.get_reward(), done, {'is_success': True}

#
# import os
# print(os.getcwd())
# from stable_baselines.common.env_checker import check_env
# envs = CitationNormal()
# print("Observation space:", envs.observation_space.shape)
# print("Action space:", envs.action_space.shape)
# check_env(envs, warn=True)
#
