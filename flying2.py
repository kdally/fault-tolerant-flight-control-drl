import os
import time
import warnings
import signal

import pandas as pd
from agent.sac import SAC
from agent.policy import LnMlpPolicy
from agent.callback import SaveOnBestReturn

from envs.citation_rates import Citation
from tools.schedule import schedule
from tools.identifier import get_ID
from tools.plot_training import plot_training
from tools.plot_response import get_response
from flying import run_preexisting
from get_task import get_task

warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')


def keyboardInterruptHandler(signal, frame):
    run_preexisting(directory='tmp_rates')
    exit(0)


def learn():

    env_train = Citation(task=get_task()[:3], time_vector=get_task()[3])
    env_eval = Citation(task=get_task()[:3], time_vector=get_task()[3])

    callback = SaveOnBestReturn(eval_env=env_eval, eval_freq=2000, log_path="agent/trained/tmp_rates/",
                                best_model_save_path="agent/trained/tmp_rates/")
    model = SAC(LnMlpPolicy, env_train, verbose=1,
                ent_coef='auto', batch_size=256,
                learning_rate=schedule(0.0003, 0.0003)
                )
    model.learn(total_timesteps=int(2e6), log_interval=50, callback=callback)
    model = SAC.load("agent/trained/tmp_rates/best_model.zip")
    ID = get_ID(6)
    model.save(f'agent/trained/{get_task()[4]}_{ID}.zip')
    training_log = pd.read_csv('agent/trained/tmp_rates/monitor.csv')
    training_log.to_csv(f'agent/trained/{get_task()[4]}_{ID}.csv')
    plot_training(ID, get_task()[4])
    get_response(env_eval, agent=model, ID=ID)

    return


signal.signal(signal.SIGINT, keyboardInterruptHandler)
learn()
# run_preexisting()
# run_preexisting('5DVX67')

# os.system('say "your program has finished"')
