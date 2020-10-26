import warnings
import signal
import pandas as pd

from agent.sac import SAC
from agent.policy import LnMlpPolicy
from agent.callback import SaveOnBestReturn
from envs.citation import CitationNormal as Citation
from tools.get_task import AltitudeTask, AttitudeTask, BodyRateTask
from tools.schedule import schedule_kink, constant, schedule_exp
from tools.identifier import get_ID
from tools.plot_training import plot_training
from tools.plot_weights import plot_weights

warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')

task = AltitudeTask


def learn():
    env_train = Citation(task=task)
    env_eval = env_train.get_cousin()

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
    training_log.to_csv(f'agent/trained/{env_eval.task_fun()[4]}_{ID}.csv')
    plot_weights(ID, env_eval.task_fun()[4])
    plot_training(ID, env_eval.task_fun()[4])
    agent = SAC.load("agent/trained/tmp/best_model.zip", env=env_eval)
    agent.ID = ID
    agent.save(f'agent/trained/{env_eval.task_fun()[4]}_{agent.ID}.zip')
    env_eval = Citation(evaluation=True, task=task)
    env_eval.render(agent=agent)

    return


def run_preexisting(ID=None):
    env_eval = Citation(evaluation=True, task=task)

    if ID is None:
        env_eval.render()
    else:
        agent = SAC.load(f"agent/trained/{env_eval.task_fun()[4]}_{ID}.zip", env=env_eval)
        agent.ID = ID
        env_eval.render(agent=agent)


def keyboardInterruptHandler(signal, frame):
    print('')
    print('Early stopping. Getting last results...')
    run_preexisting()
    exit(0)


signal.signal(signal.SIGINT, keyboardInterruptHandler)
# learn()
run_preexisting('P7V00G')  # batch size 512, LR 0.0003 ct, buffer 5e4, size 64, train_freq=1
# run_preexisting('9VZ5VE')
# run_preexisting('EN0KMW')

# os.system('say "your program has finished"')
