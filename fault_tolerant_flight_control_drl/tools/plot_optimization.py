import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns

sns.set()


def plot_optimization(fname='report_sim2_5DoF'):

    df = pd.read_csv(f'./tests/optimization_logs/{fname}.csv')
    df['value'] = -df['value']
    df['params_lr'] = np.log10(df['params_lr'])
    df['params_buffer_size'] = np.log10(df['params_buffer_size'])

    df['params_net_arch'] = df['params_net_arch'].replace(['small', 'medium', 'big'], [16, 32, 64])

    data = [go.Parcoords(
        line=dict(color=df['value'], showscale=True,
                  colorscale='thermal', colorbar={'title': 'Return'},
                  cmin=-1000,
                  cmax=-500),
        dimensions=list([
            dict(range=[np.log10(1e-4), np.log10(1e-2)],
                 tickvals=[-4., -3., -2],
                 ticktext=['1E-4', '1E-3', '1E-2'],
                 label='Learning Rate', values=df['params_lr']),
            dict(tickvals=[256, 512], range=[200, 550],
                 label='Minibatch Size', values=df['params_batch_size']),
            dict(tickvals=[1, 50, 100], range=[0, 100],
                 label='Train Frequency', values=df['params_train_freq']),
            dict(tickvals=[np.log10(5e4), np.log10(1e6), np.log10(2e6)], ticktext=['5E4', '1E6', '2E6'],
                 range=[np.log10(2.8e4), np.log10(5e6)],
                 label='Buffer Size', values=df['params_buffer_size']),
            dict(label='Network Width', values=df['params_net_arch'],
                 tickvals=[16, 32, 64])  # , range = [1,3],)
        ])
    )]

    fig = go.FigureWidget(data=data)
    fig.update_layout(template="plotly")
    fig.write_image("./figures/fig_par_coords2.pdf")
    return
