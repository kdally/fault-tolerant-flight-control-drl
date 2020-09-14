import os
import time
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from agent.sac import SAC
from agent.policy import LnMlpPolicy
from agent.callback import SaveOnBestRewardSimple, EvalCallback as SaveOnBestReturn

from envs.citation import Citation as Citation_nl
from envs.citation_lin import Citation as Citation_l
from tools.schedule import schedule
from tools.identifier import get_ID
from tools.plot_training import plot_training
from stable_baselines.bench import Monitor


env_lin = Citation_l()
env_nl = Citation_nl()

obs_lin = env_lin.reset()
obs_nl = env_nl.reset()

action = np.array([0, -3*np.pi/180, 3*np.pi/180])
action_trim = np.array(
    [-0.024761262011031245, 1.3745996716698875e-14, -7.371050575286063e-14 ])

obs_lin = env_lin.step(action)
obs_nl = env_nl.step(action-action_trim)
print(f'State lin :', env_lin.state)
print(f'State nl :', env_nl.state)
print('')

action = np.array([0, -3*np.pi/180, 3*np.pi/180])

obs_lin = env_lin.step(action)
obs_nl = env_nl.step(action-action_trim)
print(f'State lin :', env_lin.state)
print(f'State nl :', env_nl.state)
print('')

action = np.array([0, -3*np.pi/180, 3*np.pi/180])

obs_lin = env_lin.step(action)
obs_nl = env_nl.step(action-action_trim)
print(f'State lin :', env_lin.state)
print(f'State nl :', env_nl.state)

