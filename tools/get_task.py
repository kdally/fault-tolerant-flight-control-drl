import numpy as np
import random
from abc import ABC, abstractmethod


class Task:

    def __init__(self):
        self.time_v = None
        self.state_indices = {'p': 0, 'q': 1, 'r': 2, 'V': 3, 'alpha': 4, 'beta': 5,
                              'phi': 6, 'theta': 7, 'psi': 8, 'h': 9, 'x': 10, 'y': 11}

        self.obs_indices = None
        self.track_signals = None
        self.track_indices = []
        self.signals = {}

        return

    def organize_indices(self, signals, obs_indices):

        track_signals = np.zeros(self.time_v.shape[0])
        track_indices = []
        for state in signals:
            if signals[state].shape[0] != self.time_v.shape[0]:
                signals[state] = np.append(signals[state], signals[state][-1])
            track_signals = np.vstack([track_signals, signals[state]])
            track_indices.append(int(self.state_indices[state]))
        track_signals = track_signals[1:]
        obs_indices = track_indices + obs_indices

        return track_signals, track_indices, obs_indices

    def choose_task(self, evaluation, failure, FDD):

        if failure[0] is not 'normal':
            if evaluation:
                if FDD:
                    task_fun = self.get_task_eval_FDD
                else:
                    task_fun = self.get_task_eval_fail
            else:
                task_fun = self.get_task_tr_fail

        else:
            if evaluation:
                task_fun = self.get_task_eval
            else:
                task_fun = self.get_task_tr

        return task_fun, evaluation, FDD

    @abstractmethod
    def get_task_tr(self):
        self.time_v: np.ndarray = np.arange(0, 20, 0.01)
        pass

    @abstractmethod
    def get_task_eval(self):
        self.time_v: np.ndarray = np.arange(0, 80, 0.01)
        pass

    @abstractmethod
    def get_task_tr_fail(self):
        pass

    @abstractmethod
    def get_task_eval_fail(self):
        pass

    @abstractmethod
    def get_task_eval_FDD(self):
        pass

    @abstractmethod
    def return_signals(self):
        pass


class BodyRateTask(Task):

    def get_task_tr(self):
        super(BodyRateTask, self).get_task_tr()

        self.signals['p'] = np.hstack([np.zeros(int(2.5 * self.time_v.shape[0] / self.time_v[-1].round())),
                                       5 * np.sin(self.time_v[:int(self.time_v.shape[0] * 3 / 4)] * 0.2 * np.pi * 2),
                                       # 5 * np.sin(time_v[:int(time_v.shape[0] / 4)] * 3.5 * np.pi * 0.2),
                                       # -5 * np.ones(int(2.5 * time_v.shape[0] / time_v[-1].round())),
                                       # 5 * np.ones(int(2.5 * time_v.shape[0] / time_v[-1].round())),
                                       # np.zeros(int(2.5 * time_v.shape[0] / time_v[-1].round())),
                                       ])
        self.signals['q'] = np.hstack([5 * np.sin(self.time_v[:int(self.time_v.shape[0] * 3 / 4)] * 0.2 * np.pi * 2),
                                       # 5 * np.sin(time_v[:int(time_v.shape[0] / 4)] * 3.5 * np.pi * 0.2),
                                       # -5 * np.ones(int(2.5 * time_v.shape[0] / time_v[-1].round())),
                                       # 5 * np.ones(int(2.5 * time_v.shape[0] / time_v[-1].round())),
                                       np.zeros(int(2.5 * self.time_v.shape[0] / self.time_v[-1].round())),
                                       ])

        return self.return_signals()

    def get_task_eval(self):
        super(BodyRateTask, self).get_task_eval()

        self.signals['p'] = np.hstack([np.zeros(int(2.5 * self.time_v.shape[0] / self.time_v[-1].round())),
                                       5 * np.sin(self.time_v[:int(self.time_v.shape[0] * 3 / 4)] * 0.2 * np.pi * 2),
                                       # 5 * np.sin(time_v[:int(time_v.shape[0] / 4)] * 3.5 * np.pi * 0.2),
                                       # -5 * np.ones(int(2.5 * time_v.shape[0] / time_v[-1].round())),
                                       # 5 * np.ones(int(2.5 * time_v.shape[0] / time_v[-1].round())),
                                       # np.zeros(int(2.5 * time_v.shape[0] / time_v[-1].round())),
                                       ])
        self.signals['q'] = np.hstack([5 * np.sin(self.time_v[:int(self.time_v.shape[0] * 3 / 4)] * 0.2 * np.pi * 2),
                                       # 5 * np.sin(time_v[:int(time_v.shape[0] / 4)] * 3.5 * np.pi * 0.2),
                                       # -5 * np.ones(int(2.5 * time_v.shape[0] / time_v[-1].round())),
                                       # 5 * np.ones(int(2.5 * time_v.shape[0] / time_v[-1].round())),
                                       np.zeros(int(2.5 * self.time_v.shape[0] / self.time_v[-1].round())),
                                       ])

        return self.return_signals()

    def get_task_tr_fail(self):
        raise NotImplementedError

    def get_task_eval_fail(self):
        raise NotImplementedError

    def get_task_eval_FDD(self):
        raise NotImplementedError

    def return_signals(self):
        self.signals['beta'] = np.zeros(int(self.time_v.shape[0]))
        self.obs_indices = [self.state_indices['r']]

        self.track_signals, self.track_indices, self.obs_indices = self.organize_indices(self.signals, self.obs_indices)

        return self.track_signals, self.track_indices, self.obs_indices, self.time_v, 'body'


class AttitudeTask(Task):

    def get_task_tr(self):
        super(AttitudeTask, self).get_task_tr()

        angle_theta = random.choice([20, 15, -20, -15])
        time_v = self.time_v
        self.signals['theta'] = np.hstack(
            [angle_theta * np.sin(time_v[:np.argwhere(time_v == 1.5)[0, 0]] * 0.16 * np.pi * 2),
             angle_theta * np.ones(int(3.5 * time_v.shape[0] / time_v[-1].round())),
             angle_theta * np.cos(time_v[:np.argwhere(time_v == 0.5)[0, 0]] * 0.33 * np.pi * 2),
             angle_theta / 2 * np.ones(int(4. * time_v.shape[0] / time_v[-1].round())),
             angle_theta / 2 * np.cos(time_v[:np.argwhere(time_v == 0.5)[0, 0]] * 0.47 * np.pi * 2),
             -angle_theta * np.sin(time_v[:np.argwhere(time_v == 1.5)[0, 0]] * 0.17 * np.pi * 2),
             -angle_theta * np.ones(int(3.5 * time_v.shape[0] / time_v[-1].round())),
             -angle_theta * np.cos(time_v[:np.argwhere(time_v == 0.5)[0, 0]] * 0.33 * np.pi * 2),
             -angle_theta / 2 * np.ones(int(4.5 * time_v.shape[0] / time_v[-1].round())),
             ])

        angle_phi = random.choice([45, 35, 25, -45, -35, -25])
        self.signals['phi'] = np.hstack([np.zeros(int(2 * time_v.shape[0] / time_v[-1].round())),
                                         angle_phi * np.sin(time_v[:np.argwhere(time_v == 1)[0, 0]] * 0.25 * np.pi * 2),
                                         angle_phi * np.ones(int(4 * time_v.shape[0] / time_v[-1].round())),
                                         angle_phi * np.cos(time_v[:np.argwhere(time_v == 1)[0, 0]] * 0.25 * np.pi * 2),
                                         np.zeros(int(2 * time_v.shape[0] / time_v[-1].round())),
                                         np.zeros(int(2 * time_v.shape[0] / time_v[-1].round())),
                                         -angle_phi * np.sin(
                                             time_v[:np.argwhere(time_v == 1)[0, 0]] * 0.25 * np.pi * 2),
                                         -angle_phi * np.ones(int(4 * time_v.shape[0] / time_v[-1].round())),
                                         -angle_phi * np.cos(
                                             time_v[:np.argwhere(time_v == 1)[0, 0]] * 0.25 * np.pi * 2),
                                         np.zeros(int(2 * time_v.shape[0] / time_v[-1].round())),
                                         ])

        return self.return_signals()

    def get_task_eval(self):
        super(AttitudeTask, self).get_task_eval()

        time_v = self.time_v
        self.signals['theta'] = np.hstack([20 * np.sin(time_v[:np.argwhere(time_v == 5.0)[0, 0]] * 0.05 * np.pi * 2),
                                           20 * np.ones(int(8 * time_v.shape[0] / time_v[-1].round())),
                                           20 * np.cos(time_v[:np.argwhere(time_v == 2.0)[0, 0]] * 0.08 * np.pi * 2),
                                           10 * np.ones(int(15 * time_v.shape[0] / time_v[-1].round())),
                                           10 * np.cos(time_v[:np.argwhere(time_v == 2.0)[0, 0]] * 0.12 * np.pi * 2),
                                           0 * np.ones(int(25 * time_v.shape[0] / time_v[-1].round())),
                                           -15 * np.sin(time_v[:np.argwhere(time_v == 2)[0, 0]] * 0.13 * np.pi * 2),
                                           -15 * np.ones(int(12 * time_v.shape[0] / time_v[-1].round())),
                                           -15 * np.cos(time_v[:np.argwhere(time_v == 4)[0, 0]] * 0.06 * np.pi * 2),
                                           0 * np.ones(int(5 * time_v.shape[0] / time_v[-1].round())),
                                           ])
        sign = 1
        self.signals['phi'] = np.hstack([0 * np.ones(int(5 * time_v.shape[0] / time_v[-1].round())),
                                         sign * 45 * np.sin(
                                             time_v[:np.argwhere(time_v == 2.0)[0, 0]] * 0.13 * np.pi * 2),
                                         sign * 45 * np.ones(int(9 * time_v.shape[0] / time_v[-1].round())),
                                         sign * 45 * np.cos(time_v[:np.argwhere(time_v == 2)[0, 0]] * 0.12 * np.pi * 2),
                                         0 * np.ones(int(4 * time_v.shape[0] / time_v[-1].round())),
                                         -sign * 30 * np.sin(
                                             time_v[:np.argwhere(time_v == 2)[0, 0]] * 0.13 * np.pi * 2),
                                         -sign * 30 * np.ones(int(4 * time_v.shape[0] / time_v[-1].round())),
                                         -sign * 30 * np.cos(
                                             time_v[:np.argwhere(time_v == 2.0)[0, 0]] * 0.12 * np.pi * 2),
                                         0 * np.ones(int(5 * time_v.shape[0] / time_v[-1].round())),
                                         sign * 70 * np.sin(
                                             time_v[:np.argwhere(time_v == 5.5)[0, 0]] * 0.05 * np.pi * 2),
                                         sign * 70 * np.ones(int(8 * time_v.shape[0] / time_v[-1].round())),
                                         sign * 70 * np.cos(
                                             time_v[:np.argwhere(time_v == 5.5)[0, 0]] * 0.04 * np.pi * 2),
                                         0 * np.ones(int(6 * time_v.shape[0] / time_v[-1].round())),
                                         -sign * 35 * np.sin(
                                             time_v[:np.argwhere(time_v == 1.5)[0, 0]] * 0.16 * np.pi * 2),
                                         -sign * 35 * np.ones(int(9 * time_v.shape[0] / time_v[-1].round())),
                                         -sign * 35 * np.cos(
                                             time_v[:np.argwhere(time_v == 1.5)[0, 0]] * 0.16 * np.pi * 2),
                                         0 * np.ones(int(8 * time_v.shape[0] / time_v[-1].round())),
                                         ])

        return self.return_signals()

    def get_task_tr_fail(self, theta_angle=15):
        super(AttitudeTask, self).get_task_tr()

        time_v = self.time_v
        angle_theta = random.choice([1, -1]) * theta_angle
        self.signals['theta'] = np.hstack(
            [angle_theta * np.sin(time_v[:np.argwhere(time_v == 1.5)[0, 0]] * 0.16 * np.pi * 2),
             angle_theta * np.ones(int(6 * time_v.shape[0] / time_v[-1].round())),
             angle_theta * np.cos(time_v[:np.argwhere(time_v == 1.5)[0, 0]] * 0.16 * np.pi * 2),
             -angle_theta * np.sin(time_v[:np.argwhere(time_v == 1.5)[0, 0]] * 0.17 * np.pi * 2),
             -angle_theta * np.ones(int(5.5 * time_v.shape[0] / time_v[-1].round())),
             -angle_theta * np.cos(time_v[:np.argwhere(time_v == 1.5)[0, 0]] * 0.16 * np.pi * 2),
             np.zeros(int(2.5 * time_v.shape[0] / time_v[-1].round())),
             ])

        angle_phi = random.choice([25, -25, -30])
        self.signals['phi'] = np.hstack([np.zeros(int(1.5 * time_v.shape[0] / time_v[-1].round())),
                                         angle_phi * np.sin(time_v[:np.argwhere(time_v == 1)[0, 0]] * 0.25 * np.pi * 2),
                                         angle_phi * np.ones(int(4 * time_v.shape[0] / time_v[-1].round())),
                                         angle_phi * np.cos(time_v[:np.argwhere(time_v == 1)[0, 0]] * 0.25 * np.pi * 2),
                                         np.zeros(int(3 * time_v.shape[0] / time_v[-1].round())),
                                         -angle_phi * np.sin(
                                             time_v[:np.argwhere(time_v == 1)[0, 0]] * 0.25 * np.pi * 2),
                                         -angle_phi * np.ones(int(3.5 * time_v.shape[0] / time_v[-1].round())),
                                         -angle_phi * np.cos(
                                             time_v[:np.argwhere(time_v == 1)[0, 0]] * 0.25 * np.pi * 2),
                                         np.zeros(int(4 * time_v.shape[0] / time_v[-1].round())),
                                         ])

        return self.return_signals()

    def get_task_eval_fail(self, theta_angle=15):
        time_v: np.ndarray = np.arange(0, 70, 0.01)

        self.signals['theta'] = np.hstack([np.zeros(int(10 * time_v.shape[0] / time_v[-1].round())),
                                           theta_angle * np.sin(
                                               time_v[:np.argwhere(time_v == 4.0)[0, 0]] * 0.06 * np.pi * 2),
                                           theta_angle * np.ones(int(14 * time_v.shape[0] / time_v[-1].round())),
                                           theta_angle * np.cos(
                                               time_v[:np.argwhere(time_v == 2.0)[0, 0]] * 0.12 * np.pi * 2),
                                           0 * np.ones(int(6 * time_v.shape[0] / time_v[-1].round())),
                                           -theta_angle * np.sin(
                                               time_v[:np.argwhere(time_v == 2)[0, 0]] * 0.13 * np.pi * 2),
                                           -theta_angle * np.ones(int(14 * time_v.shape[0] / time_v[-1].round())),
                                           -theta_angle * np.cos(
                                               time_v[:np.argwhere(time_v == 4)[0, 0]] * 0.06 * np.pi * 2),
                                           0 * np.ones(int(14 * time_v.shape[0] / time_v[-1].round())),
                                           ])

        self.signals['phi'] = np.hstack([np.zeros(int(16 * time_v.shape[0] / time_v[-1].round())),
                                         -20 * np.sin(time_v[:np.argwhere(time_v == 1.5)[0, 0]] * 0.16 * np.pi * 2),
                                         -20 * np.ones(int(6 * time_v.shape[0] / time_v[-1].round())),
                                         -20 * np.cos(time_v[:np.argwhere(time_v == 1.5)[0, 0]] * 0.16 * np.pi * 2),
                                         0 * np.ones(int(16 * time_v.shape[0] / time_v[-1].round())),
                                         20 * np.sin(time_v[:np.argwhere(time_v == 1.5)[0, 0]] * 0.16 * np.pi * 2),
                                         20 * np.ones(int(6 * time_v.shape[0] / time_v[-1].round())),
                                         20 * np.cos(time_v[:np.argwhere(time_v == 1.5)[0, 0]] * 0.16 * np.pi * 2),
                                         0 * np.ones(int(20 * time_v.shape[0] / time_v[-1].round())),
                                         ])

        return self.return_signals()

    def get_task_eval_FDD(self, theta_angle=15):
        time_v = np.arange(0, 120, 0.01)
        self.signals['theta'] = np.hstack([np.zeros(int(10 * time_v.shape[0] / time_v[-1].round())),
                                           theta_angle * np.sin(
                                               time_v[:np.argwhere(time_v == 4.0)[0, 0]] * 0.06 * np.pi * 2),
                                           theta_angle * np.ones(int(14 * time_v.shape[0] / time_v[-1].round())),
                                           theta_angle * np.cos(
                                               time_v[:np.argwhere(time_v == 2.0)[0, 0]] * 0.12 * np.pi * 2),
                                           0 * np.ones(int(6 * time_v.shape[0] / time_v[-1].round())),
                                           -theta_angle * np.sin(
                                               time_v[:np.argwhere(time_v == 2)[0, 0]] * 0.13 * np.pi * 2),
                                           -theta_angle * np.ones(int(14 * time_v.shape[0] / time_v[-1].round())),
                                           -theta_angle * np.cos(
                                               time_v[:np.argwhere(time_v == 4)[0, 0]] * 0.06 * np.pi * 2),
                                           0 * np.ones(int(4 * time_v.shape[0] / time_v[-1].round())),

                                           np.zeros(int(4 * time_v.shape[0] / time_v[-1].round())),
                                           theta_angle * np.sin(
                                               time_v[:np.argwhere(time_v == 4.0)[0, 0]] * 0.06 * np.pi * 2),
                                           theta_angle * np.ones(int(14 * time_v.shape[0] / time_v[-1].round())),
                                           theta_angle * np.cos(
                                               time_v[:np.argwhere(time_v == 2.0)[0, 0]] * 0.12 * np.pi * 2),
                                           0 * np.ones(int(6 * time_v.shape[0] / time_v[-1].round())),
                                           -theta_angle * np.sin(
                                               time_v[:np.argwhere(time_v == 2)[0, 0]] * 0.13 * np.pi * 2),
                                           -theta_angle * np.ones(int(14 * time_v.shape[0] / time_v[-1].round())),
                                           -theta_angle * np.cos(
                                               time_v[:np.argwhere(time_v == 4)[0, 0]] * 0.06 * np.pi * 2),
                                           0 * np.ones(int(10 * time_v.shape[0] / time_v[-1].round())),
                                           ])

        roll_angle = 20
        self.signals['phi'] = np.hstack([np.zeros(int(16 * time_v.shape[0] / time_v[-1].round())),
                                         -roll_angle * np.sin(
                                             time_v[:np.argwhere(time_v == 1.5)[0, 0]] * 0.16 * np.pi * 2),
                                         -roll_angle * np.ones(int(6 * time_v.shape[0] / time_v[-1].round())),
                                         -roll_angle * np.cos(
                                             time_v[:np.argwhere(time_v == 1.5)[0, 0]] * 0.16 * np.pi * 2),
                                         0 * np.ones(int(16 * time_v.shape[0] / time_v[-1].round())),
                                         roll_angle * np.sin(
                                             time_v[:np.argwhere(time_v == 1.5)[0, 0]] * 0.16 * np.pi * 2),
                                         roll_angle * np.ones(int(6 * time_v.shape[0] / time_v[-1].round())),
                                         roll_angle * np.cos(
                                             time_v[:np.argwhere(time_v == 1.5)[0, 0]] * 0.16 * np.pi * 2),
                                         0 * np.ones(int(10 * time_v.shape[0] / time_v[-1].round())),

                                         np.zeros(int(10 * time_v.shape[0] / time_v[-1].round())),
                                         -roll_angle * np.sin(
                                             time_v[:np.argwhere(time_v == 1.5)[0, 0]] * 0.16 * np.pi * 2),
                                         -roll_angle * np.ones(int(6 * time_v.shape[0] / time_v[-1].round())),
                                         -roll_angle * np.cos(
                                             time_v[:np.argwhere(time_v == 1.5)[0, 0]] * 0.16 * np.pi * 2),
                                         0 * np.ones(int(16 * time_v.shape[0] / time_v[-1].round())),
                                         roll_angle * np.sin(
                                             time_v[:np.argwhere(time_v == 1.5)[0, 0]] * 0.16 * np.pi * 2),
                                         roll_angle * np.ones(int(6 * time_v.shape[0] / time_v[-1].round())),
                                         roll_angle * np.cos(
                                             time_v[:np.argwhere(time_v == 1.5)[0, 0]] * 0.16 * np.pi * 2),
                                         0 * np.ones(int(16 * time_v.shape[0] / time_v[-1].round())),
                                         ])

        return self.return_signals()

    def return_signals(self):
        self.signals['beta'] = np.zeros(int(self.time_v.shape[0]))
        self.obs_indices = [self.state_indices['p'], self.state_indices['q'], self.state_indices['r']]

        self.track_signals, self.track_indices, self.obs_indices = self.organize_indices(self.signals, self.obs_indices)

        return self.track_signals, self.track_indices, self.obs_indices, self.time_v, '3attitude_step'


class AltitudeTask(Task):

    def get_task_tr(self):
        super(AltitudeTask, self).get_task_tr()

        time_v = np.arange(0, 20, 0.01)

        self.signals['h'] = np.hstack([np.linspace(2000, 2050, int(7.5 * time_v.shape[0] / time_v[-1].round())),
                                       2050 * np.ones(int(2.5 * time_v.shape[0] / time_v[-1].round())),
                                       np.linspace(2050, 2025, int(3.75 * time_v.shape[0] / time_v[-1].round())),
                                       2025 * np.ones(int(6.25 * time_v.shape[0] / time_v[-1].round())),
                                       ])

        angle_phi = random.choice([45, 35, -45, -35])
        self.signals['phi'] = np.hstack([np.zeros(int(1 * time_v.shape[0] / time_v[-1].round())),
                                         angle_phi * np.sin(time_v[:np.argwhere(time_v == 1)[0, 0]] * 0.25 * np.pi * 2),
                                         angle_phi * np.ones(int(4 * time_v.shape[0] / time_v[-1].round())),
                                         angle_phi * np.cos(time_v[:np.argwhere(time_v == 1)[0, 0]] * 0.25 * np.pi * 2),
                                         np.zeros(int(3 * time_v.shape[0] / time_v[-1].round())),
                                         np.zeros(int(2 * time_v.shape[0] / time_v[-1].round())),
                                         -angle_phi * 0.8 * np.sin(
                                             time_v[:np.argwhere(time_v == 1)[0, 0]] * 0.25 * np.pi * 2),
                                         -angle_phi * 0.8 * np.ones(int(4 * time_v.shape[0] / time_v[-1].round())),
                                         -angle_phi * 0.8 * np.cos(
                                             time_v[:np.argwhere(time_v == 1)[0, 0]] * 0.25 * np.pi * 2),
                                         np.zeros(int(2 * time_v.shape[0] / time_v[-1].round())),
                                         ])

        return self.return_signals()

    def get_task_eval(self):
        self.time_v = time_v = np.arange(0, 200, 0.01)
        self.signals['h'] = np.hstack([2000 * np.ones(int(5 * time_v.shape[0] / time_v[-1].round())),
                                       np.linspace(2000, 2800, int(135 * time_v.shape[0] / time_v[-1].round())),
                                       2800 * np.ones(int(20 * time_v.shape[0] / time_v[-1].round())),
                                       np.linspace(2800, 2650, int(30 * time_v.shape[0] / time_v[-1].round())),
                                       2650 * np.ones(int(10 * time_v.shape[0] / time_v[-1].round())),
                                       ])
        sign = 1
        angle1 = 40
        angle2 = 30
        self.signals['phi'] = np.hstack([0 * np.ones(int(15 * time_v.shape[0] / time_v[-1].round())),
                                         sign * angle2 * np.sin(
                                             time_v[:np.argwhere(time_v == 2.0)[0, 0]] * 0.13 * np.pi * 2),
                                         sign * angle2 * np.ones(int(24 * time_v.shape[0] / time_v[-1].round())),
                                         sign * angle2 * np.cos(
                                             time_v[:np.argwhere(time_v == 2)[0, 0]] * 0.12 * np.pi * 2),
                                         0 * np.ones(int(17 * time_v.shape[0] / time_v[-1].round())),
                                         -sign * angle1 * np.sin(
                                             time_v[:np.argwhere(time_v == 2)[0, 0]] * 0.13 * np.pi * 2),
                                         -sign * angle1 * np.ones(int(27 * time_v.shape[0] / time_v[-1].round())),
                                         -sign * angle1 * np.cos(
                                             time_v[:np.argwhere(time_v == 2.0)[0, 0]] * 0.12 * np.pi * 2),
                                         0 * np.ones(int(10 * time_v.shape[0] / time_v[-1].round())),
                                         sign * angle1 * np.sin(
                                             time_v[:np.argwhere(time_v == 5.5)[0, 0]] * 0.05 * np.pi * 2),
                                         sign * angle1 * np.ones(int(23 * time_v.shape[0] / time_v[-1].round())),
                                         sign * angle1 * np.cos(
                                             time_v[:np.argwhere(time_v == 5.5)[0, 0]] * 0.045 * np.pi * 2),
                                         0 * np.ones(int(28 * time_v.shape[0] / time_v[-1].round())),
                                         -sign * angle2 * np.sin(
                                             time_v[:np.argwhere(time_v == 1.5)[0, 0]] * 0.16 * np.pi * 2),
                                         -sign * angle2 * np.ones(int(19 * time_v.shape[0] / time_v[-1].round())),
                                         -sign * angle2 * np.cos(
                                             time_v[:np.argwhere(time_v == 1.5)[0, 0]] * 0.16 * np.pi * 2),
                                         0 * np.ones(int(15 * time_v.shape[0] / time_v[-1].round())),
                                         ])

        return self.return_signals()

    def get_task_tr_fail(self):
        raise NotImplementedError

    def get_task_eval_fail(self):
        raise NotImplementedError

    def get_task_eval_FDD(self):
        raise NotImplementedError

    def return_signals(self):
        self.signals['beta'] = np.zeros(int(self.time_v.shape[0]))
        self.obs_indices = [self.state_indices['p'], self.state_indices['q'], self.state_indices['r'],
                            self.state_indices['theta']]

        self.track_signals, self.track_indices, self.obs_indices = self.organize_indices(self.signals, self.obs_indices)

        return self.track_signals, self.track_indices, self.obs_indices, self.time_v, 'altitude_2attitude'

#
# import matplotlib.pyplot as plt
#
# time_v: np.ndarray = np.arange(0, 200, 0.01)
# sig1 = np.hstack([2000 * np.ones(int(5 * time_v.shape[0] / time_v[-1].round())),
#                           np.linspace(2000, 2800, int(135 * time_v.shape[0] / time_v[-1].round())),
#                           2800 * np.ones(int(20 * time_v.shape[0] / time_v[-1].round())),
#                           np.linspace(2800, 2650, int(30 * time_v.shape[0] / time_v[-1].round())),
#                           2650 * np.ones(int(10 * time_v.shape[0] / time_v[-1].round())),
#                           ])
# sign = 1
# angle1 = 40
# angle2 = 30
# sig2 = np.hstack([0 * np.ones(int(15 * time_v.shape[0] / time_v[-1].round())),
#                             sign * angle2 * np.sin(time_v[:np.argwhere(time_v == 2.0)[0, 0]] * 0.13 * np.pi * 2),
#                             sign * angle2 * np.ones(int(24 * time_v.shape[0] / time_v[-1].round())),
#                             sign * angle2 * np.cos(time_v[:np.argwhere(time_v == 2)[0, 0]] * 0.12 * np.pi * 2),
#                             0 * np.ones(int(17 * time_v.shape[0] / time_v[-1].round())),
#                             -sign * angle1 * np.sin(time_v[:np.argwhere(time_v == 2)[0, 0]] * 0.13 * np.pi * 2),
#                             -sign * angle1 * np.ones(int(27 * time_v.shape[0] / time_v[-1].round())),
#                             -sign * angle1 * np.cos(time_v[:np.argwhere(time_v == 2.0)[0, 0]] * 0.12 * np.pi * 2),
#                             0 * np.ones(int(10 * time_v.shape[0] / time_v[-1].round())),
#                             sign * angle1 * np.sin(time_v[:np.argwhere(time_v == 5.5)[0, 0]] * 0.05 * np.pi * 2),
#                             sign * angle1 * np.ones(int(23 * time_v.shape[0] / time_v[-1].round())),
#                             sign * angle1 * np.cos(time_v[:np.argwhere(time_v == 5.5)[0, 0]] * 0.045 * np.pi * 2),
#                             0 * np.ones(int(28 * time_v.shape[0] / time_v[-1].round())),
#                             -sign * angle2 * np.sin(time_v[:np.argwhere(time_v == 1.5)[0, 0]] * 0.16 * np.pi * 2),
#                             -sign * angle2 * np.ones(int(19 * time_v.shape[0] / time_v[-1].round())),
#                             -sign * angle2 * np.cos(time_v[:np.argwhere(time_v == 1.5)[0, 0]] * 0.16 * np.pi * 2),
#                             0 * np.ones(int(15 * time_v.shape[0] / time_v[-1].round())),
#                             ])
#
# plt.plot(time_v, sig1/50)
# plt.plot(time_v, sig2)
# plt.show()
