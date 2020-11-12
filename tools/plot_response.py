import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# todo: reorganize plotting function


def plot_response(name, env, task, perf, during_training=False, failure=None, FDD=False, broken=False):

    subplot_indices = {0: [1, 2], 1: [1, 1], 3: [2, 2], 4: [2, 1], 5: [4, 2],
                       6: [3, 2], 7: [3, 1], 8: [7, 1], 9: [5, 1], 10: [7, 2], 11: [7, 2]}

    fig = make_subplots(rows=6, cols=2)

    if broken:
        env.time = env.time[:env.step_count-2]
        env.state_history = env.state_history[:env.step_count-2]

    if failure != 'normal' and not during_training:
        fig.add_shape(
            dict(type="line", xref="x1", yref="paper",
                 x0=5, y0=0, x1=5, y1=1, line=dict(color="Grey", width=1)))
        fig.add_shape(
            dict(type="line", xref="x2", yref="paper",
                 x0=5, y0=0, x1=5, y1=1, line=dict(color="Grey", width=1)),
        )

    if FDD:
        fig.add_shape(
            dict(type="line", xref="x1", yref="paper",
                 x0=env.FDD_switch_time, y0=0, x1=env.FDD_switch_time, y1=1, line=dict(color="Grey", width=1, dash='dash')))
        fig.add_shape(
            dict(type="line", xref="x2", yref="paper",
                 x0=env.FDD_switch_time, y0=0, x1=env.FDD_switch_time, y1=1, line=dict(color="Grey", width=1, dash='dash')),
        )

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
        fig.update_yaxes(title_text='&#916;h [m]', row=4, col=1, title_standoff=8)

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
        fig.update_yaxes(title_text='&#916;h [m]', row=4, col=1, title_standoff=8)

    fig.append_trace(go.Scatter(
        x=env.time, y=env.state_history[0, :].T, name=r'$p [^\circ/s]$',
        line=dict(color='#636EFA')), row=1, col=2)
    fig.update_yaxes(title_text='p [&deg;/s]', row=1, col=2, title_standoff=0)

    fig.append_trace(go.Scatter(
        x=env.time, y=env.state_history[1, :].T, name=r'$q [^\circ/s]$',
        line=dict(color='#636EFA')), row=1, col=1)
    fig.update_yaxes(title_text='q [&deg;/s]', row=1, col=1)

    # fig.append_trace(go.Scatter(
    #     x=env.time, y=env.state_history[2, :].T, name=r'$r [^\circ/s]$',
    #     line=dict(color='#636EFA')), row=2, col=2)
    # fig.update_yaxes(title_text='r [&deg;/s]', row=2, col=2, title_standoff=6)

    fig.append_trace(go.Scatter(
        x=env.time, y=env.state_history[3, :].T, name=r'$V [m/s]$',
        line=dict(color='#636EFA')), row=2, col=2)
    fig.update_yaxes(title_text='V [m/s]', row=2, col=2, title_standoff=23)

    fig.append_trace(go.Scatter(
        x=env.time, y=env.state_history[4, :].T, name=r'$\alpha [^\circ]$',
        line=dict(color='#636EFA')), row=2, col=1)
    fig.update_yaxes(title_text='&#945; [&deg;]', row=2, col=1)

    fig.append_trace(go.Scatter(
        x=env.time, y=env.state_history[5, :].T, name=r'$\beta [^\circ]$',
        line=dict(color='#636EFA')), row=4, col=2)
    fig.update_yaxes(title_text='&#946; [&deg;]', row=4, col=2,
                     # range=[-1, 1],
                     title_standoff=0)

    fig.append_trace(go.Scatter(
        x=env.time, y=env.state_history[6, :].T, name=r'$\phi [^\circ]$',
        line=dict(color='#636EFA')), row=3, col=2)
    fig.update_yaxes(title_text='&#966; [&deg;]', row=3, col=2, title_standoff=16)
    fig.append_trace(go.Scatter(
        x=env.time, y=env.state_history[7, :].T, name=r'$\theta [^\circ]$',
        line=dict(color='#636EFA')), row=3, col=1)
    fig.update_yaxes(title_text='&#952; [&deg;]', row=3, col=1)

    fig.append_trace(go.Scatter(
        x=env.time, y=env.state_history[9, :].T, name=r'$h [m]$',
        line=dict(color='#636EFA')), row=5, col=1)
    fig.update_yaxes(title_text='h [m]', row=5, col=1, title_standoff=8)

    fig.append_trace(go.Scatter(
        x=env.time, y=env.action_history[0, :].T,
        name=r'$\delta_e [^\circ]$', line=dict(color='#00CC96')), row=6, col=1)
    fig.update_yaxes(title_text='&#948;<sub>e</sub> [&deg;]', row=6, col=1)
    fig.append_trace(go.Scatter(
        x=env.time, y=env.action_history[1, :].T,
        name='&#948; [&deg;]', line=dict(color='#00CC96')), row=5, col=2)
    fig.update_yaxes(title_text='&#948;<sub>a</sub> [&deg;]', row=5, col=2, title_standoff=5)
    fig.append_trace(go.Scatter(
        x=env.time, y=env.action_history[2, :].T,
        name=r'$\delta_r [^\circ]$', line=dict(color='#00CC96')), row=6, col=2)
    fig.update_yaxes(title_text='&#948;<sub>r</sub> [&deg;]', row=6, col=2, title_standoff=5)

    fig.update_layout(showlegend=False, width=800, height=500, margin=dict(
        l=10,
        r=10,
        b=10,
        t=10,
    ))

    end_time = env.time[-1] + env.dt * 2

    if 9 in task[1]:
        tick_interval = 40
    else:
        tick_interval = 10

    fig.update_xaxes(title_text="Time [s]", range=[0, end_time], tickmode='array',
                     tickvals=np.arange(0, end_time, tick_interval), row=6, col=1)
    fig.update_xaxes(title_text="Time [s]", range=[0, end_time], tickmode='array',
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

