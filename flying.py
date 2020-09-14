import os
import time
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from agent.sac import SAC
from agent.policy import LnMlpPolicy
from agent.callback import SaveOnBestRewardSimple, EvalCallback as SaveOnBestReturn

from envs.citation import Citation
from tools.schedule import schedule
from tools.identifier import get_ID
from tools.plot_training import plot_training
from stable_baselines.bench import Monitor

warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')


def get_task(time_v: np.ndarray = np.arange(0, 10, 0.05)):
    state_indices = {'p': 0, 'q': 1, 'r': 2, 'V': 3, 'alpha': 4, 'beta': 5,
                     'phi': 6, 'theta': 7, 'psi': 8, 'h': 9, 'x': 10, 'y': 11}
    # noinspection PyDictCreation
    signals = {}

    task_type = 'body_rates'
    # task_type = '3attitude'
    # task_type = 'altitude_2attitude'

    if task_type == 'body_rates':
        signals['p'] = np.hstack([np.zeros(int(2.5 * time_v.shape[0] / time_v[-1].round())),
                                  5 * np.sin(time_v[:int(time_v.shape[0] * 3 / 4)] * 3 * np.pi * 0.2),
                                  # 5 * np.sin(time_v[:int(time_v.shape[0] / 4)] * 3.5 * np.pi * 0.2),
                                  # -5 * np.ones(int(2.5 * time_v.shape[0] / time_v[-1].round())),
                                  # 5 * np.ones(int(2.5 * time_v.shape[0] / time_v[-1].round())),
                                  # np.zeros(int(2.5 * time_v.shape[0] / time_v[-1].round())),
                                  ])
        signals['q'] = np.hstack([5 * np.sin(time_v[:int(time_v.shape[0] * 3 / 4)] * 3 * np.pi * 0.2),
                                  # 5 * np.sin(time_v[:int(time_v.shape[0] / 4)] * 3.5 * np.pi * 0.2),
                                  # -5 * np.ones(int(2.5 * time_v.shape[0] / time_v[-1].round())),
                                  # 5 * np.ones(int(2.5 * time_v.shape[0] / time_v[-1].round())),
                                  np.zeros(int(2.5 * time_v.shape[0] / time_v[-1].round())),
                                  ])
        signals['beta'] = np.zeros(int(time_v.shape[0]))
        obs_indices = [state_indices['r']]

    elif task_type == '3attitude':
        signals['theta'] = np.hstack([7 * np.sin(time_v[:int(time_v.shape[0] / 3)] * 1.5 * np.pi * 0.2),
                                  7 * np.sin(time_v[:int(time_v.shape[0] / 3)] * 3 * np.pi * 0.2),
                                  # -5 * np.ones(int(2.5 * time_v.shape[0] / time_v[-1].round())),
                                  # 5 * np.ones(int(2.5 * time_v.shape[0] / time_v[-1].round())),
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

    track_signals = np.zeros(time_v.shape[0])
    track_indices = []
    for state in signals:
        track_signals = np.vstack([track_signals, signals[state]])
        track_indices.append(int(state_indices[state]))
    track_signals = track_signals[1:]
    obs_indices = track_indices + obs_indices

    return track_signals, track_indices, obs_indices, time_v, task_type


def plot_response(name, env, task, perf):

    subplot_indices = {0: [1, 2], 1: [1, 1], 2: [2, 2], 3: [4, 1], 4: [2, 1], 5: [4, 2],
                     6: [3, 2], 7: [3, 1], 8: [7, 1], 9: [5, 1], 10: [7, 2], 11: [7, 2]}

    fig = make_subplots(rows=6, cols=2)

    for sig_index, state_index in enumerate(task[1]):
        fig.append_trace(go.Scatter(
            x=env.time, y=env.ref_signal[sig_index, :],
            line=dict(color='#EF553B', dash='dashdot')),
            row=subplot_indices[state_index][0], col=subplot_indices[state_index][1])

    fig.append_trace(go.Scatter(
        x=env.time, y=env.state_history[0, :].T, name=r'$p [^\circ/s]$',
        line=dict(color='#636EFA')), row=1, col=2)
    fig.update_yaxes(title_text=r'$p \:[^\circ/s]$', row=1, col=2)

    fig.append_trace(go.Scatter(
        x=env.time, y=env.state_history[1, :].T, name=r'$q [^\circ/s]$',
        line=dict(color='#636EFA')), row=1, col=1)
    fig.update_yaxes(title_text=r'$q \:[^\circ/s]$', row=1, col=1)

    fig.append_trace(go.Scatter(
        x=env.time, y=env.state_history[2, :].T, name=r'$r [^\circ/s]$',
        line=dict(color='#636EFA')), row=2, col=2)
    fig.update_yaxes(title_text=r'$r \:[^\circ/s]$', row=2, col=2)

    fig.append_trace(go.Scatter(
        x=env.time, y=env.state_history[3, :].T, name=r'$V [m/s]$',
        line=dict(color='#636EFA')), row=4, col=1)
    fig.update_yaxes(title_text=r'$V \:[m/s]$', row=4, col=1, title_standoff=0)

    fig.append_trace(go.Scatter(
        x=env.time, y=env.state_history[4, :].T, name=r'$\alpha [^\circ]$',
        line=dict(color='#636EFA')), row=2, col=1)
    fig.update_yaxes(title_text=r'$\alpha \:[^\circ]$', row=2, col=1)

    fig.append_trace(go.Scatter(
        x=env.time, y=env.state_history[5, :].T, name=r'$\beta [^\circ]$',
        line=dict(color='#636EFA')), row=4, col=2)
    fig.update_yaxes(title_text=r'$\beta \:[^\circ]$', row=4, col=2, range=[-0.5, 0.5])

    fig.append_trace(go.Scatter(
        x=env.time, y=env.state_history[6, :].T, name=r'$\phi [^\circ]$',
        line=dict(color='#636EFA')), row=3, col=2)
    fig.update_yaxes(title_text=r'$\phi \:[^\circ]$', row=3, col=2)
    fig.append_trace(go.Scatter(
        x=env.time, y=env.state_history[7, :].T, name=r'$\theta [^\circ]$',
        line=dict(color='#636EFA')), row=3, col=1)
    fig.update_yaxes(title_text=r'$\theta \:[^\circ]$', row=3, col=1)

    fig.append_trace(go.Scatter(
        x=env.time, y=env.state_history[9, :].T, name=r'$h [m]$',
        line=dict(color='#636EFA')), row=5, col=1)
    fig.update_yaxes(title_text=r'$h \:[m]$', row=5, col=1)

    fig.append_trace(go.Scatter(
        x=env.time, y=env.action_history[0, :].T,
        name=r'$\delta_e [^\circ]$', line=dict(color='#00CC96')), row=6, col=1)
    fig.update_yaxes(title_text=r'$\delta_e \:[^\circ]$', row=6, col=1)
    fig.append_trace(go.Scatter(
        x=env.time, y=env.action_history[1, :].T,
        name=r'$\delta_a [^\circ]$', line=dict(color='#00CC96')), row=5, col=2)
    fig.update_yaxes(title_text=r'$\delta_a \:[^\circ]$', row=5, col=2)
    fig.append_trace(go.Scatter(
        x=env.time, y=env.action_history[2, :].T,
        name=r'$\delta_r [^\circ]$', line=dict(color='#00CC96')), row=6, col=2)
    fig.update_yaxes(title_text=r'$\delta_r \:[^\circ]$', row=6, col=2)

    fig.update_layout(showlegend=False, width=800, height=500, margin=dict(
        l=10,
        r=10,
        b=10,
        t=10,
    ))

    end_time = env.time[-1] + env.dt * 2
    fig.update_xaxes(title_text="Time [s]", range=[0, end_time], tickmode='array',
                     tickvals=np.arange(0, end_time, 5), row=6, col=1)
    fig.update_xaxes(title_text="Time [s]", range=[0, end_time], tickmode='array',
                     tickvals=np.arange(0, end_time, 5), row=6, col=2)

    for row in range(6):
        for col in range(3):
            fig.update_xaxes(showticklabels=False, nticks=7, row=row, col=col)

    fig.update_traces(mode='lines')
    fig.write_image(f"figures/{get_task()[4]}_{name}_r{abs(int(perf))}.eps")


env_train = Citation(task=get_task()[:3], time_vector=get_task()[3])
env_eval = Citation(task=get_task()[:3], time_vector=get_task()[3])

learn = False

if learn:
    # todo: choose and study callback functions!
    callback = SaveOnBestReturn(eval_env=env_eval, eval_freq=1000, log_path="agent/trained/tmp/",
                                best_model_save_path="agent/trained/tmp/", n_eval_episodes=5)

    # env_train = Monitor(env_train, "agent/trained/tmp")
    # callback = SaveOnBestRewardSimple(check_freq=5000, log_dir="agent/trained/tmp/")

    model = SAC(LnMlpPolicy, env_train, verbose=1,
                ent_coef='auto', batch_size=256, learning_rate=schedule(0.00094))

    # model = SAC.load("tmp/best_model.zip", env=env_train)
    tic = time.time()
    model.learn(total_timesteps=int(1e6), log_interval=50, callback=callback)
    model = SAC.load("agent/trained/tmp/best_model.zip")
    ID = get_ID(6)
    model.save(f'agent/trained/{get_task()[4]}_{ID}.zip')
    training_log = pd.read_csv('agent/trained/tmp/monitor.csv')
    training_log.to_csv(f'agent/trained/{get_task()[4]}_{ID}.csv')
    plot_training(ID, get_task()[4])
    print('')
    print(f'Elapsed time = {time.time() - tic}s')
    print('')

else:
    ID = '4Dnfdu'
    # ID = 'tmp/best_model'
    model = SAC.load(f"agent/trained/{get_task()[4]}_{ID}.zip")

obs = env_eval.reset()
return_a = 0

for i, current_time in enumerate(env_eval.time):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env_eval.step(action)
    return_a += reward
    if current_time == env_eval.time[-1]:
        plot_response(ID, env_eval, get_task(), return_a)
        print(f"Goal reached! Return = {return_a}")
        print('')
        break

os.system('say "your program has finished"')
