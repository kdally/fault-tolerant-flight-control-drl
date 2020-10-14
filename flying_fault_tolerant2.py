import warnings
import signal

import pandas as pd
from agent.sac import SAC
from agent.policy import LnMlpPolicy
from agent.callback import SaveOnBestReturn
from envs.citation import Citation

from tools.schedule import schedule_kink
from tools.identifier import get_ID
from tools.plot_weights import plot_weights
from tools.plot_training import plot_training
from tools.plot_response import get_response
from tools.get_task import get_task_tr_fail

warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')

# failure_inputs = ['de', 20.05, 3.0]
failure_inputs = ['da', 1.0, 0.3]
# failure_inputs = ['dr', 0.0, -15.0]
# failure_inputs = ['cg', 1.0, 1.04]
# failure_inputs = ['ice', 1.0, 0.7] # https://doi.org/10.1016/S0376-0421(01)00018-5
# failure_inputs = ['ht', 1.0, 1.0]
# failure_inputs = ['vt', 1.0, 0.0]


def learn():

    env_train = Citation(failure=failure_inputs)
    env_eval = Citation(failure=failure_inputs)

    callback = SaveOnBestReturn(eval_env=env_eval, eval_freq=2000, log_path="agent/trained/tmp/",
                                best_model_save_path="agent/trained/tmp/")

    agent = SAC(LnMlpPolicy, env_train, verbose=1,
                ent_coef='auto', batch_size=256,
                learning_rate=schedule_kink(0.0004, 0.0002),
                )
    agent.learn(total_timesteps=int(6e3), log_interval=50, callback=callback)
    ID = get_ID(6) + f'_{failure_inputs[0]}'
    plot_weights(agent.weights_sample, ID, get_task_tr_fail()[4])
    agent = SAC.load("agent/trained/tmp/best_model.zip")
    agent.save(f'agent/trained/{get_task_tr_fail()[4]}_{ID}.zip')
    training_log = pd.read_csv('agent/trained/tmp/monitor.csv')
    training_log.to_csv(f'agent/trained/{get_task_tr_fail()[4]}_{ID}.csv')
    # plot_training(ID, get_task_tr_fail()[4])
    # get_response(Citation(evaluation=True, failure=failure_inputs), agent=agent, ID=ID, failure=True)
    return


def run_preexisting(ID=None, directory: str = 'tmp'):

    env_eval = Citation(evaluation=True, failure=failure_inputs)

    if ID is None:
        agent = SAC.load(f"agent/trained/{directory}/best_model.zip")
        get_response(env_eval, agent=agent, failure=True)
    else:
        agent = SAC.load(f"agent/trained/{get_task_tr_fail()[4]}_{ID}.zip")
        get_response(env_eval, agent=agent, ID=ID+f'_{failure_inputs[0]}', failure=True)


def keyboardInterruptHandler(signal, frame):
    print('')
    print('Early stopping. Getting last results...')
    run_preexisting()
    exit(0)


signal.signal(signal.SIGINT, keyboardInterruptHandler)
learn()
# run_preexisting('9VZ5VE') # general, robust
# run_preexisting('PBUGP7_ht')
# run_preexisting('last')

# os.system('say "your program has finished"')
