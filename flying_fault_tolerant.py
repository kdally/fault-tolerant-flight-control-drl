import warnings
import signal

import pandas as pd
import numpy as np
from agent.sac import SAC
from agent.policy import LnMlpPolicy
from agent.callback import SaveOnBestReturn
from envs.citation import CitationIcing as Citation

from tools.schedule import schedule_kink, constant
from tools.identifier import get_ID
from tools.plot_training import plot_training
from tools.plot_weights import plot_weights
from tools.get_task import get_task_tr_fail


warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')

# failure_inputs = ['de', 20.05, 3.0]
# failure_inputs = ['da', 1.0, 0.3]
# failure_inputs = ['dr', 0.0, -15.0]
# failure_inputs = ['cg', 1.0, 1.04]
failure_inputs = ['ice', 1.0, 0.7] # https://doi.org/10.1016/S0376-0421(01)00018-5
# failure_inputs = ['ht', 1.0, 0.5]
# failure_inputs = ['vt', 1.0, 0.0]


def learn():

    env_train = Citation()
    env_eval = Citation()

    callback = SaveOnBestReturn(eval_env=env_eval, eval_freq=2000, log_path="agent/trained/tmp/",
                                best_model_save_path="agent/trained/tmp/")

    agent = SAC(LnMlpPolicy, env_train, verbose=1,
                ent_coef='auto', batch_size=512,
                learning_rate=constant(0.0003),
                train_freq=100,
                policy_kwargs=dict(layers=[32, 32]),
                )
    # agent = SAC.load(f"agent/trained/{get_task_tr_fail()[4]}_9VZ5VE.zip", env=env_train)
    agent.learn(total_timesteps=int(2e6), log_interval=50, callback=callback)
    agent.ID = get_ID(6) + f'_{failure_inputs[0]}'
    training_log = pd.read_csv('agent/trained/tmp/monitor.csv')
    training_log.to_csv(f'agent/trained/{get_task_tr_fail()[4]}_{agent.ID}.csv')
    plot_weights(agent.ID, get_task_tr_fail()[4])
    plot_training(agent.ID, get_task_tr_fail()[4])
    agent = SAC.load("agent/trained/tmp/best_model.zip", env=env_eval)
    agent.save(f'agent/trained/{get_task_tr_fail()[4]}_{agent.ID}.zip')
    env_eval = Citation(evaluation=True)
    env_eval.render(agent=agent)

    return


def run_preexisting(ID=None):

    env_eval = Citation(evaluation=True)

    if ID is None:
        env_eval.render()
    else:
        agent = SAC.load(f"agent/trained/{get_task_tr_fail()[4]}_{ID}.zip", env=env_eval)
        agent.ID = ID
        env_eval.render(agent=agent)


def keyboardInterruptHandler(signal, frame):
    print('')
    print('Early stopping. Getting last results...')
    run_preexisting()
    exit(0)


signal.signal(signal.SIGINT, keyboardInterruptHandler)
learn()
# run_preexisting('9VZ5VE') # general, robust
# run_preexisting('R0EV0V_ht')

# run_preexisting('last')

# os.system('say "your program has finished"')
