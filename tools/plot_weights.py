import numpy as np
import pandas as pd
import plotly.graph_objects as go
pd.set_option('mode.chained_assignment', None)


def plot_training(ID: str, task_type: str):

    df = pd.read_csv(f'/Users/kdally/OneDrive - Delft University of Technology/TU/MSc '
                     f'Thesis/DRL-cessna-citation-fc/agent/trained/{task_type}_{ID}.csv', header=0)

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


def plot_trainings(IDs: list, task_type: str, avg_param: int = 2):

    df = pd.read_csv(f'/Users/kdally/OneDrive - Delft University of Technology/TU/MSc '
                     f'Thesis/DRL-cessna-citation-fc/agent/trained/{task_type}_{IDs[0]}.csv', header=0)

    for i in range(len(IDs)):

        df_extra = pd.read_csv(f'/Users/kdally/OneDrive - Delft University of Technology/TU/MSc '
                         f'Thesis/DRL-cessna-citation-fc/agent/trained/{task_type}_{IDs[i]}.csv', header=0)
        assert df_extra.shape[0] == df.shape[0]

        df[f'r_{i}'] = df_extra['r']

    df['r_avg_vert'] = df.iloc[: , 4:4+len(IDs)].mean(axis=1)
    df['r_avg_horz'] = df['r_avg_vert']
    df['r_min'] = df.iloc[:, 4:4+len(IDs)].min(axis=1)
    df['r_max'] = df.iloc[:, 4:4+len(IDs)].max(axis=1)

    for i in range(1, df.shape[0]):

        lower_index = max(int(i - avg_param), 1)
        df['r_avg_horz'][i] = df['r_avg_vert'][lower_index:i + 1].mean().copy()

    df['r_avg_horz'] = -np.log10(-df['r_avg_horz'])
    df['r_min'] = -np.log10(-df['r_min'])
    df['r_max'] = -np.log10(-df['r_max'])
    return_ticks = np.array([-2000, -1000, -500, -250, -125, -60, -30, -15])
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=np.hstack([df['l'].to_numpy(),df['l'][::-1].to_numpy()]),
        y=np.hstack([df['r_max'].to_numpy(), df['r_min'][::-1].to_numpy()]),
        fill='toself',
        fillcolor='rgba(99,110,250,0.3)',
        line=dict(color='rgba(255, 255, 255, 0)'),
        showlegend=False,
    ))

    fig.add_trace(go.Scatter(
        x=df['l'], y=df['r_avg_horz'],
        showlegend=False,
        line=dict(color='#636EFA'),
    ))

    fig.update_layout(
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
            tickfont=dict(size=18)
        ),
        xaxis_title='Training steps', yaxis_title='Return',
        template="plotly", height=400,
        margin=dict(l=50, r=5, b=50, t=10)
    )
    fig.update_traces(mode='lines')
    fig.write_image(f"/Users/kdally/OneDrive - Delft University of Technology/TU/MSc "
                    f"Thesis/DRL-cessna-citation-fc/figures/{task_type}_training.pdf")


# plot_training('MDl9EX', 'body_rates')
# plot_training('PBUGP7_ht', '3attitude_step')
# plot_trainings(['CQJTZA', 'PDB9SI'], '3attitude')