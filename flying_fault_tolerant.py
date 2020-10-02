import warnings
import signal

import pandas as pd
from agent.sac import SAC
from agent.policy import LnMlpPolicy
from agent.callback import SaveOnBestReturn
from envs.citation_test import Citation

from tools.schedule import schedule_kink
from tools.identifier import get_ID
from tools.plot_training import plot_training
from tools.plot_response import get_response
from tools.get_task import get_task_tr_fail

warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')

failure_type = 'cg'  # de, da, dr5, dr15, struc, cg

if failure_type == 'de':
    import envs.elevatorrange._citation as C_MODEL
elif failure_type == 'da':
    import envs.aileroneff._citation as C_MODEL
elif failure_type == 'dr5':
    import envs.rudderstuck5._citation as C_MODEL
elif failure_type == 'dr15':
    import envs.rudderstuck15._citation as C_MODEL
elif failure_type == 'struc':
    import envs.structural._citation as C_MODEL
elif failure_type == 'cg':
    import envs.cgshift._citation as C_MODEL
else:
    raise ValueError(f"Failure type not recognized.")


def learn():

    env_train = Citation(C_MODEL, failure=True)
    env_eval = Citation(C_MODEL, failure=True)

    callback = SaveOnBestReturn(eval_env=env_eval, eval_freq=2000, log_path="agent/trained/tmp/",
                                best_model_save_path="agent/trained/tmp/")

    agent = SAC(LnMlpPolicy, env_train, verbose=1,
                ent_coef='auto', batch_size=256,
                learning_rate=schedule_kink(0.0004, 0.0002),
                )
    agent.learn(total_timesteps=int(1e6), log_interval=50, callback=callback)
    agent = SAC.load("agent/trained/tmp/best_model.zip")
    ID = get_ID(6) + f'_{failure_type}'
    agent.save(f'agent/trained/{get_task_tr_fail()[4]}_{ID}.zip')
    training_log = pd.read_csv('agent/trained/tmp/monitor.csv')
    training_log.to_csv(f'agent/trained/{get_task_tr_fail()[4]}_{ID}.csv')
    plot_training(ID, get_task_tr_fail()[4])
    get_response(Citation(C_MODEL, eval=True, failure=True), agent=agent, ID=ID)

    return


def run_preexisting(ID=None, directory: str = 'tmp'):

    env_eval = Citation(C_MODEL, eval=True, failure=True)

    if ID is None:
        agent = SAC.load(f"agent/trained/{directory}/best_model.zip")
        get_response(env_eval, agent=agent)
    else:
        agent = SAC.load(f"agent/trained/{get_task_tr_fail()[4]}_{ID}.zip")
        get_response(env_eval, agent=agent, ID=ID+f'_{failure_type}')


def keyboardInterruptHandler(signal, frame):
    print('')
    print('Early stopping. Getting last results...')
    run_preexisting()
    exit(0)


signal.signal(signal.SIGINT, keyboardInterruptHandler)
# learn()
run_preexisting('9VZ5VE')
# run_preexisting('last_nosidesliptracking')


# os.system('say "your program has finished"')
