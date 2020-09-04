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

sys.path.append('envs/lin/save_mat.m')

log_dir = "agents/tmp"
os.makedirs(log_dir, exist_ok=True)
env = Citation()

learn = False

if learn:
    env = Monitor(env, log_dir)
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
controller_1 = PID(Kp=10, Ki=0.5, Kd=1)
controller_2 = PID(Kp=10, Ki=0.5, Kd=1)
controller_3 = PID(Kp=10, Ki=0.5, Kd=1)

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
        print("Goal reached!", "return=", return_a)

        # fig = go.Figure()
        fig = make_subplots(
            rows=3, cols=1,
            specs=[[{"rowspan": 2}],
                   [None],
                   [{}]])

        fig.append_trace(go.Scatter(
            x=env.time, y=env.state_history[1, :].T, name=r'$x_c\: (SAC)$',
            line=dict(color='#636EFA')), row=1, col=1)

        fig.append_trace(go.Scatter(
            x=env.time, y=env.ref_cart, name=r'$x_{ref}$',
            line=dict(color='#EF553B')), row=1, col=1)
        fig.append_trace(go.Scatter(
            x=env.time, y=env.state_history[4, :].T, name=r'$\theta\: (SAC)$',
            line=dict(color='#636EFA', dash='dashdot')), row=1, col=1)

        fig.append_trace(go.Scatter(
            x=env.time, y=env.ref_ball, name=r'$\theta_{ref}$',
            line=dict(color='#EF553B', dash='dashdot')), row=1, col=1)

        fig.append_trace(go.Scatter(
            x=env.time, y=env.action_history[0, :].T,
            name=r'$F_1 \: (SAC)$', line=dict(color='#ab63f3')), row=3, col=1)
        fig.append_trace(go.Scatter(
            x=env.time, y=env.action_history[1, :].T,
            name=r'$F_2 \: (SAC)$', line=dict(color='#19d3f3')), row=3, col=1)

        fig.update_yaxes(title_text="Position [m], [rad]", range=[-1, 7],
                         tickfont=dict(size=15), row=1, col=1, title_standoff=50)

        fig.update_layout(width=600, height=400, margin=dict(
            l=10,
            r=10,
            b=10,
            t=10,
        ))

        end_time = env.time[-1] + env.dt * 2
        fig.update_xaxes(range=[0, end_time], tickmode='array',
                         tickvals=np.arange(0, end_time, 5), tickfont=dict(size=15), row=1, col=1)
        fig.update_xaxes(title_text="Time [s]", range=[0, end_time], tickmode='array',
                         tickvals=np.arange(0, end_time, 5), tickfont=dict(size=15), row=3, col=1)
        fig.update_yaxes(title_text="Force [N]", range=[-1100, 1100], tickmode='array',
                         tickvals=np.linspace(-1000, 1000, 3), tickfont=dict(size=15), row=3, col=1)

        fig.update_layout(font=dict(size=13), template="plotly", legend=dict(font=dict(size=16)))
        fig.update_traces(mode='lines')

        fig.write_image(f"figures/flying_{abs(int(return_a))}.eps")

        print(f"Goal reached! Return = {return_a}")
        print('')
        break

os.system('say "your program has finished"')
