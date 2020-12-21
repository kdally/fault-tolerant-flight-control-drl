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
    fig.update_yaxes(title_text=r'$p\:\: [\frac{\text{deg}}{\text{s}}]$', row=1, col=2, title_standoff=7,
                     tickfont=dict(size=11),
                     titlefont=dict(size=13),
                     )

    fig.append_trace(go.Scatter(
        x=time_vec, y=env.state_history[1, :].T, name=r'$q [^\circ/s]$',
        line=dict(color='#636EFA')), row=1, col=1)
    fig.update_yaxes(title_text=r'$q\:\: [\frac{\text{deg}}{\text{s}}]$', row=1, col=1, title_standoff=13,
                     # tickmode='array',
                     # tickvals=np.arange(-5, 5+2.5, 2.5),
                     # ticktext=['-5',' ', '0',' ', '5'],
                     # range=[-5, 6],
                     tickfont=dict(size=11),
                     titlefont=dict(size=13)
                     )

    fig.append_trace(go.Scatter(
        x=time_vec, y=env.state_history[2, :].T, name=r'$r [^\circ/s]$',
        line=dict(color='#636EFA')), row=2, col=2)
    fig.update_yaxes(row=2, col=2, title_standoff=14,
                                      tickmode='array',
                                      title_text=r'$r\:\: [\frac{\text{deg}}{s}]$',
                                      tickfont=dict(size=11),
                                      titlefont=dict(size=13)
                                      )

    fig.append_trace(go.Scatter(
        x=time_vec, y=env.state_history[3, :].T, name=r'$V [m/s]$',
        line=dict(color='#636EFA')), row=4, col=1)
    fig.update_yaxes(title_text=r'$V\:\: [\frac{\text{m}}{\text{s}}]$', row=4, col=1, title_standoff=13,
                     # tickmode='array',
                     # tickvals=np.arange(88, 90+1, 1),
                     # ticktext=['88', '89', '90'],
                     tickfont=dict(size=11),
                     # range=[87,90.5],
                     titlefont=dict(size=13)
                     )

    fig.append_trace(go.Scatter(
        x=time_vec, y=env.state_history[4, :].T, name=r'$\alpha [^\circ]$',
        line=dict(color='#636EFA')), row=2, col=1)
    fig.update_yaxes(title_text=r'$\alpha\:\: [\text{deg}]$', row=2, col=1, title_standoff=18,
                     # tickmode='array',
                     # tickvals=np.arange(2, 6+1, 1),
                     # ticktext=['2', ' ','4', ' ', '6'],
                     # range=[1.5, 6],
                     tickfont=dict(size=11),
                     titlefont=dict(size=13)
                     )

    fig.append_trace(go.Scatter(
        x=time_vec, y=env.state_history[5, :].T, name=r'$\beta [^\circ]$',
        line=dict(color='#636EFA')), row=4, col=2)
    fig.update_yaxes(title_text=r'$\beta\:\: [\text{deg}]$', row=4, col=2, title_standoff=14,
                     # tickmode='array',
                     # tickvals=np.arange(-1, 1 + 0.5, 0.5),
                     # ticktext=['-1', ' ', '0', ' ', '1'],
                     # range=[-1, 1],
                     tickfont=dict(size=11),
                     titlefont=dict(size=13)
                     )

    fig.append_trace(go.Scatter(
        x=time_vec, y=env.state_history[6, :].T, name=r'$\phi [^\circ]$',
        line=dict(color='#636EFA')), row=3, col=2)
    fig.update_yaxes(title_text=r'$\phi\:\: [\text{deg}]$', row=3, col=2, title_standoff=6,
                     # tickmode='array',
                     # tickvals=np.arange(-40, 40 + 20, 20),
                     # ticktext=['-40', ' ', '0', ' ', '40'],
                     tickfont=dict(size=11),
                     # range=[-22, 40],
                     titlefont=dict(size=13)
                     )
    fig.append_trace(go.Scatter(
        x=time_vec, y=env.state_history[7, :].T, name=r'$\theta [^\circ]$',
        line=dict(color='#636EFA')), row=3, col=1)
    fig.update_yaxes(title_text=r'$\theta\:\: [\text{deg}]$', row=3, col=1,
                     # tickmode='array',
                     # tickvals=np.arange(0, 10 + 2.5, 2.5),
                     # ticktext=['0', ' ', '5 ', ' ', '10'],
                     tickfont=dict(size=11),
                     # range=[-16, 20.5],
                     titlefont=dict(size=13)
                     )

    fig.append_trace(go.Scatter(
        x=time_vec, y=env.state_history[9, :].T, name=r'$h [m]$',
        line=dict(color='#636EFA')), row=5, col=1)
    fig.update_yaxes(title_text=r'$h\:\: [\text{m}]$', row=5, col=1, title_standoff=5,
                     # tickmode='array',
                     # tickvals=np.arange(2000, 2400 + 100, 100),
                     # ticktext=['2000', ' ', '2200 ', ' ', '2400'],
                     tickfont=dict(size=11),
                     # range=[1980, 2400],
                     titlefont=dict(size=13)
                     )

    fig.append_trace(go.Scatter(
        x=time_vec, y=env.action_history[0, :].T,
        name=r'$\delta_e [^\circ]$', line=dict(color='#00CC96')), row=6, col=1)
    fig.update_yaxes(title_text=r'$\delta_\text{e} \:\:  [\text{deg}]$', row=6, col=1, title_standoff=20,
                     # tickmode='array',
                     # tickvals=np.arange(-10, 0 + 2.5, 2.5),
                     # ticktext=['-10', ' ', '-5', ' ', '0'],
                     tickfont=dict(size=11),
                     # range=[-10, 0],
                     titlefont=dict(size=13)
                     )
    fig.append_trace(go.Scatter(
        x=time_vec, y=env.action_history[1, :].T,
        name='&#948; [&deg;]', line=dict(color='#00CC96')), row=5, col=2)
    fig.update_yaxes(title_text=r'$\delta_\text{a} \:\:   [\text{deg}]$', row=5, col=2, title_standoff=8,
                     # tickmode='array',
                     # tickvals=np.arange(-5, 5 + 2.5, 2.5),
                     # ticktext=['-5', ' ', '0', ' ', '5'],
                     tickfont=dict(size=11),
                     # range=[-10, 10],
                     titlefont=dict(size=13)
                     )
    fig.append_trace(go.Scatter(
        x=time_vec, y=env.action_history[2, :].T,
        name=r'$\delta_r [^\circ]$', line=dict(color='#00CC96')), row=6, col=2)
    fig.update_yaxes(title_text=r'$\delta_\text{r} \:\: [\text{deg}]$', row=6, col=2, title_standoff=13,
                     # tickmode='array',
                     # tickvals=np.arange(0, 20 + 5, 5),
                     # ticktext=['0', ' ', '10', ' ', '20'],
                     tickfont=dict(size=11),
                     # range=[-5, 6],
                     titlefont=dict(size=13)
                     )

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
    eng.cd('/Users/kdally/OneDrive - Delft University of Technology/TU/MSc Thesis/Simulink/citast_python_verif')
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

RMSE = np.sqrt(np.mean((env_eval.state_history[:, :1001] - state_history_MATLAB.T)**2, axis=1))\
       / state_history_MATLAB.T.mean(axis=1)

print(f'The nRMSE averaged over all states is {RMSE.mean()*100}%.')
plot_response_verification('verif_shortperiod', state_history_MATLAB, env_eval, time_v)
