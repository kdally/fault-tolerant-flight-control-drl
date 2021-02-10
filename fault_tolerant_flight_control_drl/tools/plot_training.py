import numpy as np
import pandas as pd
import plotly.graph_objects as go

pd.set_option('mode.chained_assignment', None)


def plot_training(ID: str, task_type: str):
    """
    plot the training progress of one controller
    """

    df = pd.read_csv(f'fault_tolerant_flight_control_drl/agent/trained/{task_type}_{ID}.csv', header=0)

    df['r'] = -np.log10(-df['r'])
    # df['r_avg'] = -np.log10(-df['r_avg'])
    # df['r_avg'] = df['r'].ewm(alpha=0.1).mean()
    df['r_avg'] = df['r'].rolling(50, min_periods=1).mean()

    return_ticks = np.array([-2000, -1000, -500, -250, -125, -60, -30, -15])
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['l'], y=df['r'], mode='markers',
        marker=dict(
            color='rgba(99,110,250,0.6)',
        ),
    ))

    fig.add_trace(go.Scatter(
        x=df['l'], y=df['r_avg'], mode='lines', line=dict(color='darkblue'),
    ))

    fig.update_layout(
        showlegend=False,
        width=800, height=300,
        font=dict(size=17),
        yaxis=dict(
            tickmode='array',
            tickvals=-np.log10(-return_ticks),
            ticktext=return_ticks,
            # range=[-np.log10(300), -np.log10(50)],
            # range=[-500, 0],
            tickfont=dict(size=18)
        ),
        xaxis=dict(
            tickfont=dict(size=18),
            range=[0, df['l'].iloc[-1]]
        ),
        xaxis_title='Training steps', yaxis_title='Return',
        template="plotly",
        margin=dict(l=50, r=5, b=50, t=10)
    )

    fig.write_image(f"figures/{task_type}_{ID}_training.pdf")


def plot_trainings(IDs: list, task_type: str, window: int = 20):
    """
    plot the training progress of several controllers
    """

    df = pd.read_csv(f'fault_tolerant_flight_control_drl/agent/trained/{task_type}_{IDs[0]}.csv', header=0)

    for i in range(len(IDs)):
        df_extra = pd.read_csv(f'fault_tolerant_flight_control_drl/agent/trained/{task_type}_{IDs[i]}.csv', header=0)
        assert df_extra.shape[0] == df.shape[0]

        df[f'r_{i}'] = df_extra['r']

    df['r_avg'] = df.iloc[:, 4:4 + len(IDs)].mean(axis=1)
    df['r_up'] = df.iloc[:, 4:4 + len(IDs)].quantile(q=0.75, axis=1)
    df['r_down'] = df.iloc[:, 4:4 + len(IDs)].quantile(q=0.25, axis=1)

    df['r_avg_smooth'] = df['r_avg'].rolling(window=window, min_periods=1).mean()
    df['r_up_smooth'] = df['r_up'].rolling(window=window, min_periods=1).mean()
    df['r_down_smooth'] = df['r_down'].rolling(window=window, min_periods=1).mean()

    df['r_avg_smooth'] = -np.log10(-df['r_avg_smooth'])
    df['r_up_smooth'] = -np.log10(-df['r_up_smooth'])
    df['r_down_smooth'] = -np.log10(-df['r_down_smooth'])
    return_ticks = np.array([-2000, -1000, -500, -250, -125, -60, -30, -15])
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=np.hstack([df['l'].to_numpy(), df['l'][::-1].to_numpy()]),
        y=np.hstack([df['r_up_smooth'].to_numpy(), df['r_down_smooth'][::-1].to_numpy()]),
        fill='toself',
        fillcolor='rgba(99,110,250,0.3)',
        line=dict(color='rgba(255, 255, 255, 0)'),
        showlegend=False,
    ))

    fig.add_trace(go.Scatter(
        x=df['l'], y=df['r_avg_smooth'],
        showlegend=False,
        line=dict(color='#636EFA'),
    ))

    fig.update_layout(
        font=dict(size=17),
        yaxis=dict(
            tickmode='array',
            tickvals=-np.log10(-return_ticks),
            ticktext=return_ticks,
            range=[-np.log10(2000), -np.log10(200)],
            # range=[-500, 0],
            tickfont=dict(size=18)
        ),
        xaxis=dict(
            tickfont=dict(size=18),
            tickvals=np.linspace(0, 1e6, 6),
            ticktext=['0', '0.2M', '0.4M', '0.6M', '0.8M', '1M'],
            range=[0, 1e6],
        ),
        xaxis_title='Training steps', yaxis_title='Return',
        template="plotly", height=400,
        margin=dict(l=50, r=5, b=50, t=10)
    )
    fig.update_traces(mode='lines')
    fig.write_image(f"figures/{task_type}_training.pdf")


def plot_trainings_cascaded(IDs_1: list, IDs_2: list, window: int = 20):
    """
    plot the training progress of cascaded controllers
    """

    df1 = pd.read_csv(f'fault_tolerant_flight_control_drl/agent/trained/3attitude_step_{IDs_1[0]}.csv', header=0)

    for i in range(len(IDs_1)):
        df_extra1 = pd.read_csv(f'fault_tolerant_flight_control_drl/agent/trained/3attitude_step_{IDs_1[i]}.csv', header=0)
        assert df_extra1.shape[0] == df1.shape[0]

        df1[f'r_{i}'] = df_extra1['r']

    df1['r_avg'] = df1.iloc[:, 4:4 + len(IDs_1)].mean(axis=1)
    df1['r_up'] = df1.iloc[:, 4:4 + len(IDs_1)].quantile(q=0.75, axis=1)
    df1['r_down'] = df1.iloc[:, 4:4 + len(IDs_1)].quantile(q=0.25, axis=1)

    df1['r_avg_smooth'] = df1['r_avg'].rolling(window=window, min_periods=1).mean()
    df1['r_up_smooth'] = df1['r_up'].rolling(window=window, min_periods=1).mean()
    df1['r_down_smooth'] = df1['r_down'].rolling(window=window, min_periods=1).mean()
    # df1['r_up_smooth'] = df1['r_up']
    # df1['r_down_smooth'] = df1['r_down']

    df1['r_avg_smooth'] = -np.log10(-df1['r_avg_smooth'])
    df1['r_up_smooth'] = -np.log10(-df1['r_up_smooth'])
    df1['r_down_smooth'] = -np.log10(-df1['r_down_smooth'])

    df2 = pd.read_csv(f'fault_tolerant_flight_control_drl/agent/trained/altitude_2pitch_{IDs_2[0]}.csv', header=0)

    df2 = df2.iloc[:, :4]

    for i in range(len(IDs_2)):
        df_extra2 = pd.read_csv(f'fault_tolerant_flight_control_drl/agent/trained/altitude_2pitch_{IDs_2[i]}.csv', header=0)
        df2[f'r_{i}'] = df_extra2['r']

    df2['r_avg'] = df2.iloc[:, 4:4 + len(IDs_2)].mean(axis=1)
    df2['r_up'] = df2.iloc[:, 4:4 + len(IDs_2)].quantile(q=0.75, axis=1)
    df2['r_down'] = df2.iloc[:, 4:4 + len(IDs_2)].quantile(q=0.25, axis=1)

    df2['r_avg_smooth'] = df2['r_avg'].rolling(window=window, min_periods=1).mean()
    df2['r_up_smooth'] = df2['r_up'].rolling(window=window, min_periods=1).mean()
    df2['r_down_smooth'] = df2['r_down'].rolling(window=window, min_periods=1).mean()
    # df2['r_up_smooth'] = df2['r_up']
    # df2['r_down_smooth'] = df2['r_down']

    df2['r_avg_smooth'] = -np.log10(-df2['r_avg_smooth'])
    df2['r_up_smooth'] = -np.log10(-df2['r_up_smooth'])
    df2['r_down_smooth'] = -np.log10(-df2['r_down_smooth'])

    return_ticks = np.array([-2000, -1000, -500, -200, -100, -50, -30, -15])
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=np.hstack([df1['l'].to_numpy(), df1['l'][::-1].to_numpy()]),
        y=np.hstack([df1['r_up_smooth'].to_numpy(), df1['r_down_smooth'][::-1].to_numpy()]),
        fill='toself',
        fillcolor='rgba(99,110,250,0.3)',
        line=dict(color='rgba(255, 255, 255, 0)'),
        showlegend=False,
    ))

    fig.add_trace(go.Scatter(
        x=df1['l'], y=df1['r_avg_smooth'],
        showlegend=True,
        line=dict(color='#636EFA'), name='Attitude controller only'
    ))

    fig.add_trace(go.Scatter(
        x=np.hstack([df2['l'].to_numpy(), df2['l'][::-1].to_numpy()]),
        y=np.hstack([df2['r_up_smooth'].to_numpy(), df2['r_down_smooth'][::-1].to_numpy()]),
        fill='toself',
        fillcolor='rgba(228, 87, 86, 0.3)',
        line=dict(color='rgba(228, 87, 86, 0)'),
        showlegend=False,
    ))

    fig.add_trace(go.Scatter(
        x=df2['l'], y=df2['r_avg_smooth'],
        showlegend=True,
        line=dict(color='#E45756'), name='Altitude controller with trained inner-loop attitude controller'
    ))

    fig.layout.font.family = 'Arial'

    fig.update_layout(
        font=dict(size=19),
        yaxis=dict(
            tickmode='array',
            tickvals=-np.log10(-return_ticks),
            ticktext=return_ticks,
            range=[-np.log10(2000), -np.log10(70)],
            # range=[-500, 0],
            tickfont=dict(size=17),
            title_font_family='Balto',
        ),
        xaxis=dict(
            tickfont=dict(size=17),
            tickvals=np.linspace(0, 1e6, 6),
            ticktext=['0', '0.2', '0.4', '0.6', '0.8', '1'],
            range=[0, 1e6],
            title_font_family='Balto',
        ),
        xaxis_title='Training time-steps', yaxis_title='Episode sum of rewards',
        template="plotly", height=400,
        margin=dict(l=10, r=40, b=10, t=5),
        legend=dict(
            font=dict(family='Balto', size=18),
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(230,236,245,1)',
        )
    )

    fig.add_annotation(text="10<sup>6</sup>",
                       xref="paper", yref="paper",
                       x=1.065, y=-0.13, showarrow=False)

    fig.update_xaxes(title_font_family="Balto")
    fig.update_yaxes(title_font_family="Balto")

    fig.update_traces(mode='lines')
    fig.write_image(f"figures/combined_training.pdf")


def plot_trainings_sensitivity(IDs_1: list, IDs_2: list, IDs_3: list, window: int = 20):
    """
    plot the training progress of controllers for sensitivity analysis
    """

    df1 = pd.read_csv(f'fault_tolerant_flight_control_drl/agent/trained/3attitude_step_{IDs_1[0]}.csv', header=0)

    # for i in range(len(IDs_1)):
    #     df_extra1 = pd.read_csv(f'fault_tolerant_flight_control_drl/agent/trained/3attitude_step_{IDs_1[i]}.csv', header=0)
    #     assert df_extra1.shape[0] == df1.shape[0]
    #
    #     df1[f'r_{i}'] = df_extra1['r']

    df1['r_avg'] = df1['r']
    # df1['r_up'] = df1.iloc[:, 4:4 + len(IDs_1)].quantile(q=0.75, axis=1)
    # df1['r_down'] = df1.iloc[:, 4:4 + len(IDs_1)].quantile(q=0.25, axis=1)

    df1['r_avg_smooth'] = df1['r_avg'].rolling(window=window, min_periods=1).mean()
    # df1['r_up_smooth'] = df1['r_up'].rolling(window=window, min_periods=1).mean()
    # df1['r_down_smooth'] = df1['r_down'].rolling(window=window, min_periods=1).mean()
    # df1['r_up_smooth'] = df1['r_up']
    # df1['r_down_smooth'] = df1['r_down']

    # df1['r_avg_smooth'] = -np.log10(-df1['r_avg_smooth'])
    # df1['r_up_smooth'] = -np.log10(-df1['r_up_smooth'])
    # df1['r_down_smooth'] = -np.log10(-df1['r_down_smooth'])

    df2 = pd.read_csv(f'fault_tolerant_flight_control_drl/agent/trained/3attitude_step_{IDs_2[0]}.csv', header=0)

    # df2 = df2.iloc[:, :4]

    # for i in range(len(IDs_2)):
    #     df_extra2 = pd.read_csv(f'fault_tolerant_flight_control_drl/agent/trained/3attitude_step_{IDs_2[i]}.csv', header=0)
    #     df2[f'r_{i}'] = df_extra2['r']

    df2['r_avg'] = df2['r']
    # df2['r_up'] = df2.iloc[:, 4:4 + len(IDs_2)].quantile(q=0.75, axis=1)
    # df2['r_down'] = df2.iloc[:, 4:4 + len(IDs_2)].quantile(q=0.25, axis=1)

    df2['r_avg_smooth'] = df2['r_avg'].rolling(window=window, min_periods=1).mean()
    # df2['r_up_smooth'] = df2['r_up'].rolling(window=window, min_periods=1).mean()
    # df2['r_down_smooth'] = df2['r_down'].rolling(window=window, min_periods=1).mean()
    # df2['r_up_smooth'] = df2['r_up']
    # df2['r_down_smooth'] = df2['r_down']

    # df2['r_avg_smooth'] = -np.log10(-df2['r_avg_smooth'])
    # df2['r_up_smooth'] = -np.log10(-df2['r_up_smooth'])
    # df2['r_down_smooth'] = -np.log10(-df2['r_down_smooth'])

    df3 = pd.read_csv(f'fault_tolerant_flight_control_drl/agent/trained/3attitude_step_{IDs_3[0]}.csv', header=0)

    df3 = df3.iloc[:, :4]

    # for i in range(len(IDs_2)):
    #     df_extra3 = pd.read_csv(f'fault_tolerant_flight_control_drl/agent/trained/3attitude_step_{IDs_3[i]}.csv', header=0)
    #     df3[f'r_{i}'] = df_extra3['r']

    df3['r_avg'] = df3['r']
    # df3['r_up'] = df3.iloc[:, 4:4 + len(IDs_2)].quantile(q=0.75, axis=1)
    # df3['r_down'] = df3.iloc[:, 4:4 + len(IDs_2)].quantile(q=0.25, axis=1)

    df3['r_avg_smooth'] = df3['r_avg'].rolling(window=window, min_periods=1).mean()
    # df3['r_up_smooth'] = df3['r_up'].rolling(window=window, min_periods=1).mean()
    # df3['r_down_smooth'] = df3['r_down'].rolling(window=window, min_periods=1).mean()

    # df3['r_avg_smooth'] = -np.log10(-df3['r_avg_smooth'])
    # df3['r_up_smooth'] = -np.log10(-df3['r_up_smooth'])
    # df3['r_down_smooth'] = -np.log10(-df3['r_down_smooth'])

    # return_ticks = np.array([-2000, -1000, -500, -200, -100, -50, -30, -15])
    fig = go.Figure()
    #
    # fig.add_trace(go.Scatter(
    #     x=np.hstack([df1['l'].to_numpy(), df1['l'][::-1].to_numpy()]),
    #     y=np.hstack([df1['r_up_smooth'].to_numpy(), df1['r_down_smooth'][::-1].to_numpy()]),
    #     fill='toself',
    #     fillcolor='rgba(99,110,250,0.3)',
    #     line=dict(color='rgba(255, 255, 255, 0)'),
    #     showlegend=False,
    # ))

    fig.add_trace(go.Scatter(
        x=df1['l'], y=df1['r_avg_smooth'],
        showlegend=True,
        name='Absolute linear',
        # line=dict(color='#636EFA'),
    ))

    # fig.add_trace(go.Scatter(
    #     x=np.hstack([df2['l'].to_numpy(), df2['l'][::-1].to_numpy()]),
    #     y=np.hstack([df2['r_up_smooth'].to_numpy(), df2['r_down_smooth'][::-1].to_numpy()]),
    #     fill='toself',
    #     fillcolor='rgba(228, 87, 86, 0.3)',
    #     line=dict(color='rgba(228, 87, 86, 0)'),
    #     showlegend=False,
    # ))

    fig.add_trace(go.Scatter(
        x=df2['l'], y=df2['r_avg_smooth'],
        showlegend=True,
        name='Square',
        # line=dict(color='#E45756'),
    ))

    # fig.add_trace(go.Scatter(
    #     x=np.hstack([df3['l'].to_numpy(), df3['l'][::-1].to_numpy()]),
    #     y=np.hstack([df3['r_up_smooth'].to_numpy(), df3['r_down_smooth'][::-1].to_numpy()]),
    #     fill='toself',
    #     fillcolor='rgba(228, 87, 186, 0.3)',
    #     line=dict(color='rgba(228, 87, 186, 0)'),
    #     showlegend=False,
    # ))

    fig.add_trace(go.Scatter(
        x=df3['l'], y=df3['r_avg_smooth'],
        showlegend=True,
        name='Rational',
        # line=dict(color='rgba(228, 87, 186, 1)'),
    ))

    fig.layout.font.family = 'Arial'

    fig.update_layout(
        font=dict(size=19),
        yaxis=dict(
            tickmode='array',
            # tickvals=-np.log10(-return_ticks),
            # ticktext=return_ticks,
            # range=[-np.log10(2000), -np.log10(70)],
            # range=[-500, 0],
            tickfont=dict(size=17),
            title_font_family='Balto',
        ),
        xaxis=dict(
            tickfont=dict(size=17),
            tickvals=np.linspace(0, 1e6, 6),
            ticktext=['0', '0.2', '0.4', '0.6', '0.8', '1'],
            range=[0, 1e6],
            title_font_family='Balto',
        ),
        xaxis_title='Training time-steps', yaxis_title='Return',
        template="plotly", height=400,
        margin=dict(l=10, r=40, b=10, t=5),
        legend=dict(
            font=dict(family='Balto', size=20),
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(230,236,245,1)',
        )
    )

    fig.add_annotation(text="10<sup>6</sup>",
                       xref="paper", yref="paper",
                       x=1.065, y=-0.13, showarrow=False)

    fig.update_xaxes(title_font_family="Balto")
    fig.update_yaxes(title_font_family="Balto")

    fig.update_traces(mode='lines')
    fig.write_image(f"figures/combined_training.pdf")


# plot_training('PZ5QGW', 'altitude_2pitch')
# plot_training('TBNJM4', 'altitude_2pitch')
# plot_training('BZVWF5','3attitude_step')
# plot_trainings(['9VZ5VE', '8G9WIL', '0I9D1J', '7AK56O', 'GXA2KT'], '3attitude_step')
# plot_trainings(['9VZ5VE', '7AK56O','GXA2KT'], '3attitude_step')
plot_trainings_cascaded(['9VZ5VE', '8G9WIL', '0I9D1J', 'GT0PLE', 'GXA2KT'],
                        ['XQ2G4Q', 'DH0TLO', 'AZ5QGW', 'H0IC1R', 'TBNJM4'])
# plot_trainings_sensitivity(['GT0PLE'], ['BZVWF5'], ['UMJB0W'])
