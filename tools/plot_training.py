import numpy as np
import pandas as pd
import plotly.graph_objects as go

pd.set_option('mode.chained_assignment', None)


def plot_training(ID: str, task_type: str):
    df = pd.read_csv(f'/Users/kdally/OneDrive - Delft University of Technology/TU/MSc '
                     f'Thesis/DRL-cessna-citation-fc/agent/trained/{task_type}_{ID}.csv', header=0)

    print(df['r'].max())
    df['r'] = -np.log10(-df['r'])
    # df['r_avg'] = -np.log10(-df['r_avg'])
    # df['r_avg'] = df['r'].ewm(alpha=0.1).mean()
    df['r_avg'] = df['r'].rolling(50).mean()

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

    fig.write_image(f"/Users/kdally/OneDrive - Delft University of Technology/TU/MSc "
                    f"Thesis/DRL-cessna-citation-fc/figures/{task_type}_{ID}_training.pdf")


def plot_trainings(IDs: list, task_type: str, window: int = 20):
    df = pd.read_csv(f'/Users/kdally/OneDrive - Delft University of Technology/TU/MSc '
                     f'Thesis/DRL-cessna-citation-fc/agent/trained/{task_type}_{IDs[0]}.csv', header=0)

    for i in range(len(IDs)):
        df_extra = pd.read_csv(f'/Users/kdally/OneDrive - Delft University of Technology/TU/MSc '
                               f'Thesis/DRL-cessna-citation-fc/agent/trained/{task_type}_{IDs[i]}.csv', header=0)
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
    fig.write_image(f"/Users/kdally/OneDrive - Delft University of Technology/TU/MSc "
                    f"Thesis/DRL-cessna-citation-fc/figures/{task_type}_training.pdf")


def plot_trainings_cascaded(IDs_1: list, IDs_2: list, window: int = 20):
    df1 = pd.read_csv(f'/Users/kdally/OneDrive - Delft University of Technology/TU/MSc '
                      f'Thesis/DRL-cessna-citation-fc/agent/trained/3attitude_step_{IDs_1[0]}.csv', header=0)

    for i in range(len(IDs_1)):
        df_extra1 = pd.read_csv(f'/Users/kdally/OneDrive - Delft University of Technology/TU/MSc '
                                f'Thesis/DRL-cessna-citation-fc/agent/trained/3attitude_step_{IDs_1[i]}.csv', header=0)
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

    df2 = pd.read_csv(f'/Users/kdally/OneDrive - Delft University of Technology/TU/MSc '
                      f'Thesis/DRL-cessna-citation-fc/agent/trained/altitude_2pitch_{IDs_2[0]}.csv', header=0)

    df2 = df2.iloc[:, :4]

    for i in range(len(IDs_2)):
        df_extra2 = pd.read_csv(f'/Users/kdally/OneDrive - Delft University of Technology/TU/MSc '
                                f'Thesis/DRL-cessna-citation-fc/agent/trained/altitude_2pitch_{IDs_2[i]}.csv', header=0)
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

    return_ticks = np.array([-2000, -1000, -500, -250, -125, -60, -30, -15])
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
        line=dict(color='#636EFA'), name='Attitude Controller'
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
        line=dict(color='#E45756'),  name='Altitude Controller'
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

    fig.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01,
        bgcolor='rgba(0,0,0,0)',
    ))

    fig.update_traces(mode='lines')
    fig.write_image(f"/Users/kdally/OneDrive - Delft University of Technology/TU/MSc "
                    f"Thesis/DRL-cessna-citation-fc/figures/combined_training.pdf")


# plot_training('XQ2G4Q', 'altitude_2pitch')
# plot_training('R0EV0U_ht', '3attitude_step')
# plot_training('P7V00G','altitude_2attitude')
# plot_trainings(['9VZ5VE', '8G9WIL', '0I9D1J', '7AK56O', 'GXA2KT'], '3attitude_step')
# plot_trainings(['9VZ5VE', '7AK56O','GXA2KT'], '3attitude_step')
# plot_trainings_cascaded(['9VZ5VE', '8G9WIL', '0I9D1J', '7AK56O', 'GXA2KT'], ['XQ2G4Q', 'PZ5QGL'])
