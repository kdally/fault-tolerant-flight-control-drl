import numpy as np
import pandas as pd
import plotly.graph_objects as go


def plot_training(ID: str, task_type: str):

    df = pd.read_csv(f'/Users/kdally/OneDrive - Delft University of Technology/TU/MSc '
                     f'Thesis/DRL-cessna-citation-fc/agent/trained/{task_type}_{ID}.csv', header=1)

    training_steps = np.arange(0, len(df['r']) * 3000, 3000)
    df['training_steps'] = training_steps
    # df['r'] = -np.log10(-df['r'])
    # return_ticks = np.array([-50, -20, -10, -5, -2, -1, -0.5, -0.1])
    fig = go.Figure()

    # x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # x_rev = x[::-1]
    #
    # # Line 1
    # y1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # y1_upper = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    # y1_lower = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # y1_lower = y1_lower[::-1]
    # fig.add_trace(go.Scatter(
    #     x=x+x_rev,
    #     y=y1_upper+y1_lower,
    #     fill='toself',
    #     fillcolor='rgba(0,100,80,0.2)',
    #     line_color='rgba(255,255,255,0)',
    #     showlegend=False,
    #     name='Fair',
    # ))

    fig.add_trace(go.Scatter(
        x=df['training_steps'], y=df['r'],
    ))

    fig.update_layout(
        font=dict(size=17),
        yaxis=dict(
            tickmode='array',
            # tickvals=-np.log10(-return_ticks),
            # ticktext=return_ticks,
            # range=[-np.log10(300), -np.log10(50)],
            range=[-500,0],
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
    fig.write_image(f"figures/{task_type}_{ID}_training.eps")
