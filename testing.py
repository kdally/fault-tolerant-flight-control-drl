import numpy as np
import matplotlib.pyplot as plt

time_v: np.ndarray = np.arange(0, 10, 0.01)

sig = np.hstack([20 * np.sin(time_v[:np.argwhere(time_v == 1)[0, 0]] * 0.26 * np.pi * 2),
                 20 * np.ones(int(4 * time_v.shape[0] / time_v[-1].round())),
                 20 * np.cos(time_v[:np.argwhere(time_v == 0.5)[0, 0]] * 0.33 * np.pi * 2),
                 10 * np.ones(int(4.5 * time_v.shape[0] / time_v[-1].round())),
                 ])

sig2 = np.hstack([np.zeros(int(2 * time_v.shape[0] / time_v[-1].round())),
                  30 * np.sin(time_v[:np.argwhere(time_v == 1)[0, 0]] * 0.25 * np.pi * 2),
                  30 * np.ones(int(4 * time_v.shape[0] / time_v[-1].round())),
                  30 * np.cos(time_v[:np.argwhere(time_v == 1)[0, 0]] * 0.25 * np.pi * 2),
                  np.zeros(int(2 * time_v.shape[0] / time_v[-1].round())),
                  ])

plt.plot(time_v, sig)
plt.plot(time_v, sig2)
plt.show()
