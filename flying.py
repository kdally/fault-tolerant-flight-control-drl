import os
import sys
import time
import warnings

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from stable_baselines import SAC
from stable_baselines.bench import Monitor
from stable_baselines.sac.policies import LnMlpPolicy

from envs.lin.citation import Citation
from tools.PID import PID
from tools.callback import SaveOnBestTrainingRewardCallback
from tools.schedule import schedule

warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')

env = Citation()

learn = False

if learn:
    env = Monitor(env, "agents/tmp")
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
    model = SAC(LnMlpPolicy, env, verbose=1,
                ent_coef='auto', batch_size=256, learning_rate=schedule(0.00094))

    # model = SAC.load("tmp/best_model.zip", envs=envs)
    tic = time.time()
    model.learn(total_timesteps=int(5e5), log_interval=10, callback=callback)
    print('')
    print(f'Elapsed time = {time.time() - tic}s')
    print('')
    del model

# model = SAC.load("agents/500K_simple_batch256.zip")
# model = SAC.load("agents/tmp/best_model.zip")

# Test the trained agent
obs = env.reset()
controller_1 = PID(Kp=-10, Ki=0.5, Kd=1)
controller_2 = PID(Kp=-10, Ki=0.5, Kd=1)
controller_3 = PID(Kp=-10, Ki=0.5, Kd=1)

return_a = 0

for i, current_time in enumerate(env.time):
    # action, _ = model.predict(obs, deterministic=True)
    action = np.array([controller_1(env.error[0], current_time),
                       controller_2(env.error[1], current_time),
                       controller_2(env.error[2], current_time)
                       ])
    obs, reward, done, info = env.step(action)
    return_a += reward
    if current_time == env.time[-1]:

        fig = make_subplots(rows=6, cols=2,)

        fig.append_trace(go.Scatter(
            x=env.time, y=env.state_history[0, :].T, name=r'$p [^\circ/s]$',
            line=dict(color='#636EFA')), row=1, col=2)
        fig.append_trace(go.Scatter(
            x=env.time, y=env.ref_signal[0, :], name=r'$p_{ref} [^\circ/s]$',
            line=dict(color='#EF553B', dash='dashdot')), row=1, col=2)
        fig.update_yaxes(title_text=r'$p \:[^\circ/s]$', row=1, col=2)

        fig.append_trace(go.Scatter(
            x=env.time, y=env.state_history[1, :].T, name=r'$q [^\circ/s]$',
            line=dict(color='#636EFA')), row=1, col=1)
        fig.append_trace(go.Scatter(
            x=env.time, y=env.ref_signal[1, :], name=r'$q_{ref} [^\circ/s]$',
            line=dict(color='#EF553B', dash='dashdot')), row=1, col=1)
        fig.update_yaxes(title_text=r'$q \:[^\circ/s]$', row=1, col=1)

        fig.append_trace(go.Scatter(
            x=env.time, y=env.state_history[2, :].T, name=r'$r [^\circ/s]$',
            line=dict(color='#636EFA')), row=2, col=2)
        fig.append_trace(go.Scatter(
            x=env.time, y=env.ref_signal[2, :], name=r'$r_{ref} [^\circ/s]$',
            line=dict(color='#EF553B', dash='dashdot')), row=2, col=2)
        fig.update_yaxes(title_text=r'$r \:[^\circ/s]$', row=2, col=2)

        fig.append_trace(go.Scatter(
            x=env.time, y=env.state_history[3, :].T, name=r'$V [m/s]$',
            line=dict(color='#636EFA')), row=4, col=1)
        fig.update_yaxes(title_text=r'$V \:[m/s]$', row=4, col=1)

        fig.append_trace(go.Scatter(
            x=env.time, y=env.state_history[4, :].T, name=r'$\alpha [^\circ]$',
            line=dict(color='#636EFA')), row=2, col=1)
        fig.update_yaxes(title_text=r'$\alpha \:[^\circ]$', row=2, col=1)
        fig.append_trace(go.Scatter(
            x=env.time, y=env.state_history[5, :].T, name=r'$\beta [^\circ]$',
            line=dict(color='#636EFA')), row=4, col=2)
        fig.update_yaxes(title_text=r'$\beta \:[^\circ]$', row=4, col=2)
        fig.append_trace(go.Scatter(
            x=env.time, y=env.state_history[6, :].T, name=r'$\phi [^\circ]$',
            line=dict(color='#636EFA')), row=3, col=2)
        fig.update_yaxes(title_text=r'$\phi \:[^\circ]$', row=3, col=2)
        fig.append_trace(go.Scatter(
            x=env.time, y=env.state_history[7, :].T, name=r'$\theta [^\circ]$',
            line=dict(color='#636EFA')), row=3, col=1)
        fig.update_yaxes(title_text=r'$\theta \:[^\circ]$', row=3, col=1)

        fig.append_trace(go.Scatter(
            x=env.time, y=env.state_history[9, :].T, name=r'$H [m]$',
            line=dict(color='#636EFA')), row=5, col=1)
        fig.update_yaxes(title_text=r'$H \:[m]$', row=5, col=1)

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

        for i in range(6):
            for j in range(3):
                fig.update_xaxes(showticklabels=False, row=i, col=j)

        fig.update_traces(mode='lines')
        fig.write_image(f"figures/flying_{abs(int(return_a))}.eps")

        print(f"Goal reached! Return = {return_a}")
        print('')
        break

os.system('say "your program has finished"')
