import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


def plot_response(name, env, task, perf, during_training=False, failure=None, FDD=False, broken=False):

    # fig = go.Figure()
    # fig.add_trace(go.Scatter(
    #     x=env.time, y=env.ref_signal[0, :], name=r'$h [m]$',
    #     line=dict(color='#EF553B', dash='dashdot')))
    #


    subplot_indices = {0: [1, 2], 1: [1, 1], 3: [2, 2], 4: [2, 1], 5: [4, 2],
                       6: [3, 2], 7: [3, 1], 8: [7, 1], 9: [5, 1], 10: [7, 2], 11: [7, 2]}

    fig = make_subplots(rows=6, cols=2)

    if broken:
        env.time = env.time[:env.step_count-2]
        env.state_history = env.state_history[:env.step_count-2]

    if env.external_ref_signal is not None:
        fig.append_trace(go.Scatter(
            x=env.time, y=env.external_ref_signal.T, name=r'$h [m]$',
            line=dict(color='#EF553B', dash='dashdot')), row=5, col=1)

        fig.append_trace(go.Scatter(
            x=env.time, y=env.ref_signal[0, :],
            line=dict(color='#EF553B')),
            row=3, col=1)

        fig.append_trace(go.Scatter(
            x=env.time, y=env.ref_signal[1, :],
            line=dict(color='#EF553B', dash='dashdot')),
            row=3, col=2)

        fig.append_trace(go.Scatter(
            x=env.time, y=env.ref_signal[2, :],
            line=dict(color='#EF553B', dash='dashdot')),
            row=4, col=2)

        fig.append_trace(go.Scatter(
            x=env.time, y=env.state_history[9, :].T - env.external_ref_signal.T, name=r'$h [m]$',
            line=dict(color='#636EFA')), row=4, col=1)
        fig.update_yaxes(title_text=r'$\delta h \:\: [\text{m}]$', row=4, col=1, title_standoff=8)

    else:
        for sig_index, state_index in enumerate(task[1]):
            fig.append_trace(go.Scatter(
                x=env.time, y=env.ref_signal[sig_index, :],
                line=dict(color='#EF553B', dash='dashdot')),
                row=subplot_indices[state_index][0], col=subplot_indices[state_index][1])

    if env.task_fun()[4] == 'altitude_2attitude':

        fig.append_trace(go.Scatter(
            x=env.time, y=env.state_history[9, :].T-env.ref_signal[0, :], name=r'$h [m]$',
            line=dict(color='#636EFA')), row=4, col=1)
        fig.update_yaxes(title_text=r'$h\:\:  [\text{m}]$', row=4, col=1, title_standoff=8)

    fig.append_trace(go.Scatter(
        x=env.time, y=env.state_history[0, :].T, name=r'$p [\frac{deg}{s}]$',
        line=dict(color='#636EFA')), row=1, col=2)
    fig.update_yaxes(title_text=r'$p\:\: [\frac{\text{deg}}{\text{s}}]$', row=1, col=2, title_standoff=0,
                     tickfont=dict(size=10)
                     )

    fig.append_trace(go.Scatter(
        x=env.time, y=env.state_history[1, :].T, name=r'$q [^\circ/s]$',
        line=dict(color='#636EFA')), row=1, col=1)
    fig.update_yaxes(title_text=r'$q\:\: [\frac{\text{deg}}{\text{s}}]$', row=1, col=1, title_standoff=0,
                     tickmode='array',
                     tickvals=np.arange(-10, 10+5, 5),
                     ticktext=['-10',' ','0',' ','10'],
                     range=[-10, 11],
                     tickfont=dict(size=10)
                     )

    fig.append_trace(go.Scatter(
        x=env.time, y=env.state_history[2, :].T, name=r'$r [^\circ/s]$',
        line=dict(color='#636EFA')), row=2, col=2)
    fig.update_yaxes(row=2, col=2, title_standoff=0,
                     tickmode='array',
                     tickvals=np.arange(-5, 5 + 2.5, 2.5),
                     range=[-5,7],
                     ticktext=['-5', ' ', '0', ' ', '5'],
                     title_text=r'$r\:\: [\frac{\text{deg}}{s}]$',
                     tickfont=dict(size=10)
                     )

    fig.append_trace(go.Scatter(
        x=env.time, y=env.state_history[3, :].T, name=r'$V [m/s]$',
        line=dict(color='#636EFA')), row=4, col=1)
    fig.update_yaxes(title_text=r'$V\:\: [\frac{\text{m}}{\text{s}}]$', row=4, col=1, title_standoff=13,
                     tickmode='array',
                     tickvals=np.arange(80, 120+10, 10),
                     ticktext=['80', ' ', '100', ' ', '120'],
                     tickfont=dict(size=10)
                     )

    fig.append_trace(go.Scatter(
        x=env.time, y=env.state_history[4, :].T, name=r'$\alpha [^\circ]$',
        line=dict(color='#636EFA')), row=2, col=1)
    fig.update_yaxes(title_text=r'$\alpha\:\: [\text{deg}]$', row=2, col=1,
                     tickmode='array',
                     tickvals=np.arange(0, 10+5, 2.5),
                     ticktext=['0', ' ', '5', ' ', '10'],
                     range=[-2, 10],
                     tickfont=dict(size=10)
                     )

    fig.append_trace(go.Scatter(
        x=env.time, y=env.state_history[5, :].T, name=r'$\beta [^\circ]$',
        line=dict(color='#636EFA')), row=4, col=2)
    fig.update_yaxes(title_text=r'$\beta\:\: [\text{deg}]$', row=4, col=2,
                     tickmode='array',
                     tickvals=np.arange(-2, 2 + 1, 1),
                     ticktext=['-2', ' ', '0', ' ', '2'],
                     range=[-2, 2],
                     tickfont=dict(size=10),
                     title_standoff=0)

    fig.append_trace(go.Scatter(
        x=env.time, y=env.state_history[6, :].T, name=r'$\phi [^\circ]$',
        line=dict(color='#636EFA')), row=3, col=2)
    fig.update_yaxes(title_text=r'$\phi\:\: [\text{deg}]$', row=3, col=2, title_standoff=16,
                     tickmode='array',
                     tickvals=[-35,0,35,70],
                     # ticktext=['-35', '0', ' ', '70'],
                     tickfont=dict(size=10),
                     range=[-37, 73],
                     )


    fig.append_trace(go.Scatter(
        x=env.time, y=env.state_history[7, :].T, name=r'$\theta [^\circ]$',
        line=dict(color='#636EFA')), row=3, col=1)
    fig.update_yaxes(title_text=r'$\theta\:\: [\text{deg}]$', row=3, col=1)

    fig.append_trace(go.Scatter(
        x=env.time, y=env.state_history[9, :].T, name=r'$h [m]$',
        line=dict(color='#636EFA')), row=5, col=1)
    fig.update_yaxes(title_text=r'$h\:\: [\text{m}]$', row=5, col=1, title_standoff=8)

    # env.action_history = env.action_history_filtered

    fig.append_trace(go.Scatter(
        x=env.time, y=env.action_history[0, :].T,
        name=r'$\delta_e [^\circ]$', line=dict(color='#00CC96')), row=6, col=1)
    fig.update_yaxes(title_text=r'$\delta_\text{e} \:\:  [\text{deg}]$', row=6, col=1)
    fig.append_trace(go.Scatter(
        x=env.time, y=env.action_history[1, :].T,
        name='&#948; [&deg;]', line=dict(color='#00CC96')), row=5, col=2)
    fig.update_yaxes(title_text=r'$\delta_\text{a} \:\:   [\text{deg}]$', row=5, col=2, title_standoff=5)
    fig.append_trace(go.Scatter(
        x=env.time, y=env.action_history[2, :].T,
        name=r'$\delta_r [^\circ]$', line=dict(color='#00CC96')), row=6, col=2)
    fig.update_yaxes(title_text=r'$\delta_\text{r} \:\: [\text{deg}]$', row=6, col=2, title_standoff=5)

    if failure != 'normal' and not during_training:
        fig.add_vline(x=5.0, row='all', col="all", line=dict(color="Grey", width=1.5))

    if FDD:
        fig.add_vline(x=env.FDD_switch_time, row='all', col="all", line=dict(color="Grey", width=1.5, dash='dot'))

    fig.update_layout(showlegend=False, width=800, height=500, margin=dict(
        l=10,
        r=10,
        b=10,
        t=10,
    ))

    fig.layout.font.family = 'Arial'

    end_time = env.time[-1] + env.dt * 2

    if 9 in task[1]:
        tick_interval = 40
    else:
        tick_interval = 10

    fig.update_xaxes(title_text=r'$t \:\: \text{[s]}$', range=[0, end_time], tickmode='array',
                     tickvals=np.arange(0, end_time, tick_interval), row=6, col=1)
    fig.update_xaxes(title_text=r'$t \:\: \text{[s]}$', range=[0, end_time], tickmode='array',
                     tickvals=np.arange(0, end_time, tick_interval), row=6, col=2)

    for row in range(6):
        for col in range(3):
            fig.update_xaxes(showticklabels=False, tickmode='array',
                             tickvals=np.arange(0, end_time, tick_interval), row=row, col=col)

    fig.update_traces(mode='lines')
    if during_training:
        fig.write_image(f"figures/during_training/{env.task_fun()[4]}_r{abs(int(perf))}.eps")
    elif failure != 'normal':
        fig.write_image(f"figures/{name}_{failure}_r{abs(int(perf))}.pdf")
    else:
        fig.write_image(f"figures/{name}_r{abs(int(perf))}.pdf")

