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
from tools.plot_response import get_response
from tools.get_task import get_task_tr


warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')


# todo: cloud training plot +moving average


def learn():

    env_train = Citation()
    env_eval = Citation()

    callback = SaveOnBestReturn(eval_env=env_eval, eval_freq=2000, log_path="agent/trained/tmp/",
                                best_model_save_path="agent/trained/tmp/")
    agent = SAC(LnMlpPolicy, env_train, verbose=1,
                ent_coef='auto', batch_size=256,
                learning_rate=schedule_kink(0.0004, 0.0002),
                # learning_rate=schedule_exp(0.0009),
                # policy_kwargs=dict(layers=[128, 64]),
                )
    agent.learn(total_timesteps=int(5e6), log_interval=50, callback=callback)
    agent = SAC.load("agent/trained/tmp/best_model.zip")
    ID = get_ID(6)
    agent.save(f'agent/trained/{get_task_tr()[4]}_{ID}.zip')
    training_log = pd.read_csv('agent/trained/tmp/monitor.csv')
    training_log.to_csv(f'agent/trained/{get_task_tr()[4]}_{ID}.csv')
    plot_training(ID, get_task_tr()[4])
    get_response(Citation(eval=True), agent=agent, ID=ID)

    return


def run_preexisting(ID=None, directory: str = 'tmp'):

    env_eval = Citation(eval=True)

    if ID is None:
        agent = SAC.load(f"agent/trained/{directory}/best_model.zip")
        get_response(env_eval, agent=agent)
    else:
        agent = SAC.load(f"agent/trained/{get_task_tr()[4]}_{ID}.zip")
        get_response(env_eval, agent=agent, ID=ID)


def keyboardInterruptHandler(signal, frame):
    print('')
    print('Early stopping. Getting last results...')
    run_preexisting()
    exit(0)


signal.signal(signal.SIGINT, keyboardInterruptHandler)
learn()
# run_preexisting('N28KZO')
# run_preexisting('9VZ5VE')

# os.system('say "your program has finished"')
