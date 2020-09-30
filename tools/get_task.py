import math

import numpy as np
import random


def get_task_tr(time_v: np.ndarray = np.arange(0, 20, 0.01)):
    state_indices = {'p': 0, 'q': 1, 'r': 2, 'V': 3, 'alpha': 4, 'beta': 5,
                     'phi': 6, 'theta': 7, 'psi': 8, 'h': 9, 'x': 10, 'y': 11}
    signals = {}

    # task_type = 'body_rates'
    # task_type = '3attitude'
    task_type = '3attitude_step'
    # task_type = 'altitude_2attitude'

    if task_type == 'body_rates':
        signals['p'] = np.hstack([np.zeros(int(2.5 * time_v.shape[0] / time_v[-1].round())),
                                  5 * np.sin(time_v[:int(time_v.shape[0] * 3 / 4)] * 0.2 * np.pi * 2),
                                  # 5 * np.sin(time_v[:int(time_v.shape[0] / 4)] * 3.5 * np.pi * 0.2),
                                  # -5 * np.ones(int(2.5 * time_v.shape[0] / time_v[-1].round())),
                                  # 5 * np.ones(int(2.5 * time_v.shape[0] / time_v[-1].round())),
                                  # np.zeros(int(2.5 * time_v.shape[0] / time_v[-1].round())),
                                  ])
        signals['q'] = np.hstack([5 * np.sin(time_v[:int(time_v.shape[0] * 3 / 4)] * 0.2 * np.pi * 2),
                                  # 5 * np.sin(time_v[:int(time_v.shape[0] / 4)] * 3.5 * np.pi * 0.2),
                                  # -5 * np.ones(int(2.5 * time_v.shape[0] / time_v[-1].round())),
                                  # 5 * np.ones(int(2.5 * time_v.shape[0] / time_v[-1].round())),
                                  np.zeros(int(2.5 * time_v.shape[0] / time_v[-1].round())),
                                  ])
        signals['beta'] = np.zeros(int(time_v.shape[0]))
        obs_indices = [state_indices['r']]

    elif task_type == '3attitude_step':

        # angle_theta = random.choice([20, 15, -20, -15])
        # signals['theta'] = np.hstack(
        #     [angle_theta * np.sin(time_v[:np.argwhere(time_v == 1.5)[0, 0]] * 0.16 * np.pi * 2),
        #      angle_theta * np.ones(int(3.5 * time_v.shape[0] / time_v[-1].round())),
        #      angle_theta * np.cos(time_v[:np.argwhere(time_v == 0.5)[0, 0]] * 0.33 * np.pi * 2),
        #      angle_theta / 2 * np.ones(int(4.5 * time_v.shape[0] / time_v[-1].round())),
        #      -angle_theta * np.sin(time_v[:np.argwhere(time_v == 1.5)[0, 0]] * 0.17 * np.pi * 2),
        #      -angle_theta * np.ones(int(3.5 * time_v.shape[0] / time_v[-1].round())),
        #      -angle_theta * np.cos(time_v[:np.argwhere(time_v == 0.5)[0, 0]] * 0.33 * np.pi * 2),
        #      -angle_theta / 2 * np.ones(int(4.5 * time_v.shape[0] / time_v[-1].round())),
        #      ])
        #
        # angle_phi = random.choice([45, 35, 25, -45, -35, -25])
        # signals['phi'] = np.hstack([np.zeros(int(2 * time_v.shape[0] / time_v[-1].round())),
        #                             angle_phi * np.sin(time_v[:np.argwhere(time_v == 1)[0, 0]] * 0.25 * np.pi * 2),
        #                             angle_phi * np.ones(int(4 * time_v.shape[0] / time_v[-1].round())),
        #                             angle_phi * np.cos(time_v[:np.argwhere(time_v == 1)[0, 0]] * 0.25 * np.pi * 2),
        #                             np.zeros(int(2 * time_v.shape[0] / time_v[-1].round())),
        #                             np.zeros(int(2 * time_v.shape[0] / time_v[-1].round())),
        #                             -angle_phi * np.sin(time_v[:np.argwhere(time_v == 1)[0, 0]] * 0.25 * np.pi * 2),
        #                             -angle_phi * np.ones(int(4 * time_v.shape[0] / time_v[-1].round())),
        #                             -angle_phi * np.cos(time_v[:np.argwhere(time_v == 1)[0, 0]] * 0.25 * np.pi * 2),
        #                             np.zeros(int(2 * time_v.shape[0] / time_v[-1].round())),
        #                             ])

        angle_theta = random.choice([20, -20])
        signals['theta'] = np.hstack(
            [angle_theta * np.sin(time_v[:np.argwhere(time_v == 3.5)[0, 0]] * 0.07 * np.pi * 2),
             angle_theta * np.ones(int(4 * time_v.shape[0] / time_v[-1].round())),
             angle_theta * np.cos(time_v[:np.argwhere(time_v == 1.5)[0, 0]] * 0.16 * np.pi * 2),
             0 * np.ones(int(11 * time_v.shape[0] / time_v[-1].round())),
             ])

        angle_phi = random.choice([45, -45])
        sign_fun = lambda x: math.copysign(1, x)

        signals['phi'] = np.hstack([np.zeros(int(3 * time_v.shape[0] / time_v[-1].round())),
                                    angle_phi * np.sin(time_v[:np.argwhere(time_v == 1)[0, 0]] * 0.24 * np.pi * 2),
                                    angle_phi * np.ones(int(2.5 * time_v.shape[0] / time_v[-1].round())),
                                    angle_phi * np.cos(time_v[:np.argwhere(time_v == 1)[0, 0]] * 0.24 * np.pi * 2),
                                    np.zeros(int(2 * time_v.shape[0] / time_v[-1].round())),
                                    -sign_fun(angle_phi) * 70 * np.sin(
                                        time_v[:np.argwhere(time_v == 2)[0, 0]] * 0.12 * np.pi * 2),
                                    -sign_fun(angle_phi) * 70 * np.ones(
                                        int(4.5 * time_v.shape[0] / time_v[-1].round())),
                                    -sign_fun(angle_phi) * 70 * np.cos(
                                        time_v[:np.argwhere(time_v == 2)[0, 0]] * 0.12 * np.pi * 2),
                                    np.zeros(int(2 * time_v.shape[0] / time_v[-1].round())),
                                    ])

        signals['beta'] = np.zeros(int(time_v.shape[0]))
        obs_indices = [state_indices['p'], state_indices['q'], state_indices['r']]

    elif task_type == 'altitude_2attitude':
        signals['h'] = np.hstack([np.arange(0, 1000, 0.5),
                                  np.zeros(int(10 * time_v.shape[0] / time_v[-1].round())),
                                  ])
        signals['phi'] = np.hstack([5 * np.sin(time_v[:int(time_v.shape[0] / 3)] * 1.5 * np.pi * 0.2),
                                    5 * np.sin(time_v[:int(time_v.shape[0] / 3)] * 3 * np.pi * 0.2),
                                    # -5 * np.ones(int(2.5 * time_v.shape[0] / time_v[-1].round())),
                                    # 5 * np.ones(int(2.5 * time_v.shape[0] / time_v[-1].round())),
                                    np.zeros(int(10 * time_v.shape[0] / time_v[-1].round())),
                                    ])
        signals['beta'] = np.zeros(int(time_v.shape[0]))
        obs_indices = [state_indices['p'], state_indices['q'], state_indices['r']]

    else:
        raise Exception('This task has not been implemented.')

    track_signals = np.zeros(time_v.shape[0])
    track_indices = []
    for state in signals:
        if signals[state].shape[0] != time_v.shape[0]:
            signals[state] = np.append(signals[state], signals[state][-1])
        track_signals = np.vstack([track_signals, signals[state]])
        track_indices.append(int(state_indices[state]))
    track_signals = track_signals[1:]
    obs_indices = track_indices + obs_indices

    return track_signals, track_indices, obs_indices, time_v, task_type


def get_task_eval(time_v: np.ndarray = np.arange(0, 80, 0.01)):
    state_indices = {'p': 0, 'q': 1, 'r': 2, 'V': 3, 'alpha': 4, 'beta': 5,
                     'phi': 6, 'theta': 7, 'psi': 8, 'h': 9, 'x': 10, 'y': 11}
    signals = {}

    # task_type = 'body_rates'
    task_type = '3attitude_step'
    # task_type = 'altitude_2attitude'

    if task_type == 'body_rates':
        signals['p'] = np.hstack([np.zeros(int(2.5 * time_v.shape[0] / time_v[-1].round())),
                                  5 * np.sin(time_v[:int(time_v.shape[0] * 3 / 4)] * 0.2 * np.pi * 2),
                                  # 5 * np.sin(time_v[:int(time_v.shape[0] / 4)] * 3.5 * np.pi * 0.2),
                                  # -5 * np.ones(int(2.5 * time_v.shape[0] / time_v[-1].round())),
                                  # 5 * np.ones(int(2.5 * time_v.shape[0] / time_v[-1].round())),
                                  # np.zeros(int(2.5 * time_v.shape[0] / time_v[-1].round())),
                                  ])
        signals['q'] = np.hstack([5 * np.sin(time_v[:int(time_v.shape[0] * 3 / 4)] * 0.2 * np.pi * 2),
                                  # 5 * np.sin(time_v[:int(time_v.shape[0] / 4)] * 3.5 * np.pi * 0.2),
                                  # -5 * np.ones(int(2.5 * time_v.shape[0] / time_v[-1].round())),
                                  # 5 * np.ones(int(2.5 * time_v.shape[0] / time_v[-1].round())),
                                  np.zeros(int(2.5 * time_v.shape[0] / time_v[-1].round())),
                                  ])
        signals['beta'] = np.zeros(int(time_v.shape[0]))
        obs_indices = [state_indices['r']]

    elif task_type == '3attitude_step':

        # signals['theta'] = np.hstack([20 * np.sin(time_v[:np.argwhere(time_v == 5.0)[0, 0]] * 0.05 * np.pi * 2),
        #                               20 * np.ones(int(8 * time_v.shape[0] / time_v[-1].round())),
        #                               20 * np.cos(time_v[:np.argwhere(time_v == 2.0)[0, 0]] * 0.08 * np.pi * 2),
        #                               10 * np.ones(int(10 * time_v.shape[0] / time_v[-1].round())),
        #                               10 * np.cos(time_v[:np.argwhere(time_v == 2.0)[0, 0]] * 0.12 * np.pi * 2),
        #                               0 * np.ones(int(15 * time_v.shape[0] / time_v[-1].round())),
        #                               -15 * np.sin(time_v[:np.argwhere(time_v == 2)[0, 0]] * 0.13 * np.pi * 2),
        #                               -15 * np.ones(int(7 * time_v.shape[0] / time_v[-1].round())),
        #                               -15 * np.cos(time_v[:np.argwhere(time_v == 4)[0, 0]] * 0.06 * np.pi * 2),
        #                               0 * np.ones(int(5 * time_v.shape[0] / time_v[-1].round())),
        #                               ])
        # signals['phi'] = np.hstack([0 * np.ones(int(5 * time_v.shape[0] / time_v[-1].round())),
        #                             30 * np.sin(time_v[:np.argwhere(time_v == 2.0)[0, 0]] * 0.13 * np.pi * 2),
        #                             30 * np.ones(int(4 * time_v.shape[0] / time_v[-1].round())),
        #                             30 * np.cos(time_v[:np.argwhere(time_v == 2)[0, 0]] * 0.12 * np.pi * 2),
        #                             0 * np.ones(int(4 * time_v.shape[0] / time_v[-1].round())),
        #                             -30 * np.sin(time_v[:np.argwhere(time_v == 2)[0, 0]] * 0.13 * np.pi * 2),
        #                             -30 * np.ones(int(4 * time_v.shape[0] / time_v[-1].round())),
        #                             -30 * np.cos(time_v[:np.argwhere(time_v == 2.0)[0, 0]] * 0.12 * np.pi * 2),
        #                             0 * np.ones(int(5 * time_v.shape[0] / time_v[-1].round())),
        #                             45 * np.sin(time_v[:np.argwhere(time_v == 2.5)[0, 0]] * 0.1 * np.pi * 2),
        #                             45 * np.ones(int(4 * time_v.shape[0] / time_v[-1].round())),
        #                             45 * np.cos(time_v[:np.argwhere(time_v == 2.5)[0, 0]] * 0.1 * np.pi * 2),
        #                             0 * np.ones(int(6 * time_v.shape[0] / time_v[-1].round())),
        #                             -25 * np.sin(time_v[:np.argwhere(time_v == 1.5)[0, 0]] * 0.16 * np.pi * 2),
        #                             -25 * np.ones(int(4 * time_v.shape[0] / time_v[-1].round())),
        #                             -25 * np.cos(time_v[:np.argwhere(time_v == 1.5)[0, 0]] * 0.16 * np.pi * 2),
        #                             0 * np.ones(int(8 * time_v.shape[0] / time_v[-1].round())),
        #                             ])

        signals['theta'] = np.hstack([20 * np.sin(time_v[:np.argwhere(time_v == 5.0)[0, 0]] * 0.05 * np.pi * 2),
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
        signals['phi'] = np.hstack([0 * np.ones(int(5 * time_v.shape[0] / time_v[-1].round())),
                                    45 * np.sin(time_v[:np.argwhere(time_v == 2.0)[0, 0]] * 0.13 * np.pi * 2),
                                    45 * np.ones(int(9 * time_v.shape[0] / time_v[-1].round())),
                                    45 * np.cos(time_v[:np.argwhere(time_v == 2)[0, 0]] * 0.12 * np.pi * 2),
                                    0 * np.ones(int(4 * time_v.shape[0] / time_v[-1].round())),
                                    -30 * np.sin(time_v[:np.argwhere(time_v == 2)[0, 0]] * 0.13 * np.pi * 2),
                                    -30 * np.ones(int(4 * time_v.shape[0] / time_v[-1].round())),
                                    -30 * np.cos(time_v[:np.argwhere(time_v == 2.0)[0, 0]] * 0.12 * np.pi * 2),
                                    0 * np.ones(int(5 * time_v.shape[0] / time_v[-1].round())),
                                    70 * np.sin(time_v[:np.argwhere(time_v == 5.5)[0, 0]] * 0.05 * np.pi * 2),
                                    70 * np.ones(int(8 * time_v.shape[0] / time_v[-1].round())),
                                    70 * np.cos(time_v[:np.argwhere(time_v == 5.5)[0, 0]] * 0.05 * np.pi * 2),
                                    0 * np.ones(int(6 * time_v.shape[0] / time_v[-1].round())),
                                    -35 * np.sin(time_v[:np.argwhere(time_v == 1.5)[0, 0]] * 0.16 * np.pi * 2),
                                    -35 * np.ones(int(9 * time_v.shape[0] / time_v[-1].round())),
                                    -35 * np.cos(time_v[:np.argwhere(time_v == 1.5)[0, 0]] * 0.16 * np.pi * 2),
                                    0 * np.ones(int(8 * time_v.shape[0] / time_v[-1].round())),
                                    ])

        # signals['theta'] = np.zeros(int(time_v.shape[0]))
        # signals['phi'] = np.zeros(int(time_v.shape[0]))
        signals['beta'] = np.zeros(int(time_v.shape[0]))
        obs_indices = [state_indices['p'], state_indices['q'], state_indices['r']]

    elif task_type == 'altitude_2attitude':
        signals['h'] = np.hstack([np.arange(0, 1000, 0.5),
                                  np.zeros(int(10 * time_v.shape[0] / time_v[-1].round())),
                                  ])
        signals['phi'] = np.hstack([5 * np.sin(time_v[:int(time_v.shape[0] / 3)] * 1.5 * np.pi * 0.2),
                                    5 * np.sin(time_v[:int(time_v.shape[0] / 3)] * 3 * np.pi * 0.2),
                                    # -5 * np.ones(int(2.5 * time_v.shape[0] / time_v[-1].round())),
                                    # 5 * np.ones(int(2.5 * time_v.shape[0] / time_v[-1].round())),
                                    np.zeros(int(10 * time_v.shape[0] / time_v[-1].round())),
                                    ])
        signals['beta'] = np.zeros(int(time_v.shape[0]))
        obs_indices = [state_indices['p'], state_indices['q'], state_indices['r']]

    else:
        raise Exception('This task has not been implemented.')

    track_signals = np.zeros(time_v.shape[0])
    track_indices = []
    for state in signals:
        if signals[state].shape[0] != time_v.shape[0]:
            signals[state] = np.append(signals[state], signals[state][-1])
        track_signals = np.vstack([track_signals, signals[state]])
        track_indices.append(int(state_indices[state]))
    track_signals = track_signals[1:]
    obs_indices = track_indices + obs_indices

    return track_signals, track_indices, obs_indices, time_v, task_type

# import matplotlib.pyplot as plt
#
# time_v: np.ndarray = np.arange(0, 20, 0.01)
#
# angle_theta = random.choice([20, -20])
# sig1 = np.hstack([angle_theta * np.sin(time_v[:np.argwhere(time_v == 3.5)[0, 0]] * 0.07 * np.pi * 2),
#                   angle_theta * np.ones(int(4 * time_v.shape[0] / time_v[-1].round())),
#                   angle_theta * np.cos(time_v[:np.argwhere(time_v == 1.5)[0, 0]] * 0.16 * np.pi * 2),
#                   0 * np.ones(int(11 * time_v.shape[0] / time_v[-1].round())),
#                   ])
#
# angle_phi = random.choice([45, -45])
# sign_fun = lambda x: math.copysign(1, x)
#
# sig2 = np.hstack([np.zeros(int(3 * time_v.shape[0] / time_v[-1].round())),
#                   angle_phi * np.sin(time_v[:np.argwhere(time_v == 1)[0, 0]] * 0.24 * np.pi * 2),
#                   angle_phi * np.ones(int(2.5 * time_v.shape[0] / time_v[-1].round())),
#                   angle_phi * np.cos(time_v[:np.argwhere(time_v == 1)[0, 0]] * 0.24 * np.pi * 2),
#                   np.zeros(int(2 * time_v.shape[0] / time_v[-1].round())),
#                   -sign_fun(angle_phi) * 70 * np.sin(time_v[:np.argwhere(time_v == 2)[0, 0]] * 0.12 * np.pi * 2),
#                   -sign_fun(angle_phi) * 70 * np.ones(int(4.5 * time_v.shape[0] / time_v[-1].round())),
#                   -sign_fun(angle_phi) * 70 * np.cos(time_v[:np.argwhere(time_v == 2)[0, 0]] * 0.12 * np.pi * 2),
#                   np.zeros(int(2 * time_v.shape[0] / time_v[-1].round())),
#                   ])
#
# plt.plot(time_v, sig1)
# plt.plot(time_v, sig2)
# plt.show()
