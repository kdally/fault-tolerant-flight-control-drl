import matlab.engine
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from envs.citation import CitationVerif as Citation


def plot_response_verification(fig_name, state_MATLAB, env, time_vec):
    subplot_indices = {0: [1, 2], 1: [1, 1], 2: [2, 2], 3: [4, 1], 4: [2, 1], 5: [4, 2],
                       6: [3, 2], 7: [3, 1], 8: [7, 1], 9: [5, 1], 10: [7, 2], 11: [7, 2]}

    fig = make_subplots(rows=6, cols=2)

    fig.append_trace(go.Scatter(
        x=time_vec, y=env.state_history[0, :].T, name=r'$p [^\circ/s]$',
        line=dict(color='#636EFA')), row=1, col=2)
    fig.update_yaxes(title_text='p [&deg;/s]', row=1, col=2, title_standoff=0)

    fig.append_trace(go.Scatter(
        x=time_vec, y=env.state_history[1, :].T, name=r'$q [^\circ/s]$',
        line=dict(color='#636EFA')), row=1, col=1)
    fig.update_yaxes(title_text='q [&deg;/s]', row=1, col=1)

    fig.append_trace(go.Scatter(
        x=time_vec, y=env.state_history[2, :].T, name=r'$r [^\circ/s]$',
        line=dict(color='#636EFA')), row=2, col=2)
    fig.update_yaxes(title_text='r [&deg;/s]', row=2, col=2, title_standoff=6)

    fig.append_trace(go.Scatter(
        x=time_vec, y=env.state_history[3, :].T, name=r'$V [m/s]$',
        line=dict(color='#636EFA')), row=4, col=1)
    fig.update_yaxes(title_text='V [m/s]', row=4, col=1, title_standoff=23)

    fig.append_trace(go.Scatter(
        x=time_vec, y=env.state_history[4, :].T, name=r'$\alpha [^\circ]$',
        line=dict(color='#636EFA')), row=2, col=1)
    fig.update_yaxes(title_text='&#945; [&deg;]', row=2, col=1)

    fig.append_trace(go.Scatter(
        x=time_vec, y=env.state_history[5, :].T, name=r'$\beta [^\circ]$',
        line=dict(color='#636EFA')), row=4, col=2)
    fig.update_yaxes(title_text='&#946; [&deg;]', row=4, col=2,
                     # range=[-1, 1],
                     title_standoff=0)

    fig.append_trace(go.Scatter(
        x=time_vec, y=env.state_history[6, :].T, name=r'$\phi [^\circ]$',
        line=dict(color='#636EFA')), row=3, col=2)
    fig.update_yaxes(title_text='&#966; [&deg;]', row=3, col=2, title_standoff=16)
    fig.append_trace(go.Scatter(
        x=time_vec, y=env.state_history[7, :].T, name=r'$\theta [^\circ]$',
        line=dict(color='#636EFA')), row=3, col=1)
    fig.update_yaxes(title_text='&#952; [&deg;]', row=3, col=1)

    fig.append_trace(go.Scatter(
        x=time_vec, y=env.state_history[9, :].T, name=r'$h [m]$',
        line=dict(color='#636EFA')), row=5, col=1)
    fig.update_yaxes(title_text='h [m]', row=5, col=1, title_standoff=8)

    fig.append_trace(go.Scatter(
        x=time_vec, y=env.action_history[0, :].T,
        name=r'$\delta_e [^\circ]$', line=dict(color='#00CC96')), row=6, col=1)
    fig.update_yaxes(title_text='&#948;<sub>e</sub> [&deg;]', row=6, col=1)
    fig.append_trace(go.Scatter(
        x=time_vec, y=env.action_history[1, :].T,
        name='&#948; [&deg;]', line=dict(color='#00CC96')), row=5, col=2)
    fig.update_yaxes(title_text='&#948;<sub>a</sub> [&deg;]', row=5, col=2, title_standoff=5)
    fig.append_trace(go.Scatter(
        x=time_vec, y=env.action_history[2, :].T,
        name=r'$\delta_r [^\circ]$', line=dict(color='#00CC96')), row=6, col=2)
    fig.update_yaxes(title_text='&#948;<sub>r</sub> [&deg;]', row=6, col=2, title_standoff=5)

    for sig_index in range(8):
        fig.append_trace(go.Scatter(
            x=time_vec, y=state_MATLAB[:, sig_index],
            line=dict(color='#EF553B', dash='dashdot', width=2)),
            row=subplot_indices[sig_index][0], col=subplot_indices[sig_index][1])

    fig.append_trace(go.Scatter(
        x=time_vec, y=state_MATLAB[:, 9],
        line=dict(color='#EF553B', dash='dashdot', width=2)),
        row=subplot_indices[9][0], col=subplot_indices[9][1])

    fig.update_layout(showlegend=False, width=800, height=500, margin=dict(
        l=10,
        r=10,
        b=10,
        t=10,
    ))

    end_time = time_vec[-1] + time_vec * 2

    tick_interval = 2.5

    # fig.update_xaxes(title_text="Time [s]", range=[0, end_time], tickmode='array',
    #                  tickvals=np.arange(0, end_time, tick_interval), row=6, col=1)
    # fig.update_xaxes(title_text="Time [s]", range=[0, end_time], tickmode='array',
    #                  tickvals=np.arange(0, end_time, tick_interval), row=6, col=2)

    # for row in range(6):
    #     for col in range(3):
    #         fig.update_xaxes(showticklabels=False, tickmode='array',
    #                          tickvals=np.arange(0, end_time, tick_interval), row=row, col=col)

    fig.update_traces(mode='lines')
    fig.write_image(f"figures/{fig_name}.pdf")


time_v = np.arange(0, 10.01, 0.01)

# ---------------------------------
# MATLAB RESPONSE
# ---------------------------------
eng = matlab.engine.start_matlab()
try:
    eng.cd('/Users/kdally/OneDrive - Delft University of Technology/TU/MSc Thesis/Simulink/citast_python_normal')
except matlab.engine.MatlabExecutionError:
    pass
eng.verification(nargout=0)
state_history_MATLAB = np.asarray(eng.eval('outputs'))
eng.quit()
scale_s = np.ones(12)
scale_s[[0, 1, 2, 4, 5, 6, 7, 8]] = 180 / np.pi
state_history_MATLAB *= scale_s


# ---------------------------------
# PYTHON RESPONSE
# ---------------------------------
env_eval = Citation(evaluation=True)
obs = env_eval.reset()
action = np.zeros((time_v.shape[0], 3))
action[:, 0] = np.hstack([0 * np.ones(int(1 * time_v.shape[0] / time_v[-1].round())),
                          -5 * np.ones(int(2 * time_v.shape[0] / time_v[-1].round())),
                          0 * np.ones(int(7 * time_v.shape[0] / time_v[-1].round()) + 1),
                          ])
action[:, 1] = np.hstack([0 * np.ones(int(4 * time_v.shape[0] / time_v[-1].round())),
                          10 * np.ones(int(1 * time_v.shape[0] / time_v[-1].round())),
                          0 * np.ones(int(5 * time_v.shape[0] / time_v[-1].round()) + 1),
                          ])

for i, current_time in enumerate(time_v):
    obs, reward, done, info = env_eval.step(action[i, :])
    if current_time == time_v[-1]:
        break

# todo: get rmse for each variable
Error = np.sqrt(np.mean((env_eval.state_history[:, :1001]-state_history_MATLAB.T)**2, axis=1))
print(f'The RMSE error is {Error} of shape {Error.shape}')

plot_response_verification('verif_shortperiod', state_history_MATLAB, env_eval, time_v)
