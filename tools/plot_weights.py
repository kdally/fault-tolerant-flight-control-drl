import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

pd.set_option('mode.chained_assignment', None)


def plot_weights(param, ID: str, task_type: str):

    df = pd.DataFrame(param[1:])

    fig = px.line(df, x=df.index, y=df.columns)

    fig.update_layout(
        showlegend=False,
        width=800, height=300,
        font=dict(size=17),
        yaxis=dict(
            tickfont=dict(size=18)
        ),
        xaxis=dict(
            tickfont=dict(size=18),
        ),
        xaxis_title='Training steps', yaxis_title=r'$w^{[2]}$',
        template="plotly",
        margin=dict(l=50, r=5, b=50, t=10)
    )

    fig.write_image(f"/Users/kdally/OneDrive - Delft University of Technology/TU/MSc "
                    f"Thesis/DRL-cessna-citation-fc/figures/{task_type}_{ID}_weights.eps")

#
# inputs = np.array([[6,7,8],[3,8,4],[7,3,1],[6,9,5],[0,5,2]])

# plot_weights(inputs, 'MDl9EX', 'body_rates')
# plot_training('PBUGP7_ht', '3attitude_step')
# plot_trainings(['CQJTZA', 'PDB9SI'], '3attitude')
