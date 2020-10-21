import warnings
import signal

import pandas as pd
from agent.sac import SAC
from agent.policy import LnMlpPolicy
from agent.callback import SaveOnBestReturn
from envs.citation import Citation

from tools.schedule import schedule_kink, constant, schedule_exp
from tools.identifier import get_ID
from tools.plot_training import plot_training
from tools.plot_weights import plot_weights
from tools.plot_response import get_response
from tools.get_task import get_task_tr

warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')


def learn():
    env_train = Citation()
    env_eval = Citation()

    callback = SaveOnBestReturn(eval_env=env_eval, eval_freq=2000, log_path="agent/trained/tmp/",
                                best_model_save_path="agent/trained/tmp/")
    agent = SAC(LnMlpPolicy, env_train, verbose=1,
                ent_coef='auto', batch_size=512,
                # learning_rate=schedule_kink(0.0004, 0.0004),
                train_freq=100,
                learning_rate=constant(0.0004),
                policy_kwargs=dict(layers=[32, 32]),
                )
    agent.learn(total_timesteps=int(2.5e6), log_interval=50, callback=callback)
    ID = get_ID(6)
    training_log = pd.read_csv('agent/trained/tmp/monitor.csv')
    training_log.to_csv(f'agent/trained/{get_task_tr()[4]}_{ID}.csv')
    plot_weights(ID, get_task_tr()[4])
    plot_training(ID, get_task_tr()[4])
    agent = SAC.load("agent/trained/tmp/best_model.zip", env=env_eval)
    agent.save(f'agent/trained/{get_task_tr()[4]}_{ID}.zip')
    get_response(Citation(evaluation=True), agent=agent, ID=ID)

    return


def run_preexisting(ID=None, directory: str = 'tmp'):
    env_eval = Citation(evaluation=True)

    if ID is None:
        agent = SAC.load(f"agent/trained/{directory}/best_model.zip", env=env_eval)
        get_response(env_eval, agent=agent)
    else:
        agent = SAC.load(f"agent/trained/{get_task_tr()[4]}_{ID}.zip", env=env_eval)
        get_response(env_eval, agent=agent, ID=ID)


def keyboardInterruptHandler(signal, frame):
    print('')
    print('Early stopping. Getting last results...')
    run_preexisting()
    exit(0)


signal.signal(signal.SIGINT, keyboardInterruptHandler)
# learn()
run_preexisting('P7V00G') # batch size 512, LR 0.0003 ct, buffer 5e4, size 64, train_freq=1
# run_preexisting('9VZ5VE')
# run_preexisting('EN0KMW')

# os.system('say "your program has finished"')
