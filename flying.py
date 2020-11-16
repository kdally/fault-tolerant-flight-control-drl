import warnings
import signal
import pandas as pd

from agent.sac import SAC
from agent.policy import LnMlpPolicy
from agent.callback import SaveOnBestReturn
from tools.get_task import AltitudeTask, AttitudeTask, BodyRateTask, Task
from tools.schedule import schedule_kink, constant, schedule_exp
from tools.identifier import get_ID
from tools.plot_training import plot_training
from tools.plot_weights import plot_weights

warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')

from envs.citation import CitationElevRange
from envs.citation import CitationAileronEff
from envs.citation import CitationRudderStuck
from envs.citation import CitationHorzTail
from envs.citation import CitationVertTail
from envs.citation import CitationIcing
from envs.citation import CitationCgShift
from envs.citation import CitationNormal


# todo: different flight conditions


def learn(task: Task, env_type=CitationNormal):
    env_train = env_type(task=task)
    env_eval = env_type(task=task)

    callback = SaveOnBestReturn(eval_env=env_eval, eval_freq=2000, log_path="agent/trained/tmp/",
                                best_model_save_path="agent/trained/tmp/")
    agent = SAC(LnMlpPolicy, env_train, verbose=1,
                ent_coef='auto', #batch_size=512,
                learning_rate=schedule_kink(0.0005, 0.0004),
                train_freq=100,
                # learning_rate=constant(0.0003),
                policy_kwargs=dict(layers=[32, 32]),
                )
    agent.learn(total_timesteps=int(2.5e6), callback=callback)
    ID = get_ID(6)
    if env_eval.failure_input[0] != 'normal':
        ID += f'_{env_eval.failure_input[0]}'
    training_log = pd.read_csv('agent/trained/tmp/monitor.csv')
    training_log.to_csv(f'agent/trained/{env_eval.task_fun()[4]}_{ID}.csv')
    plot_weights(ID, env_eval.task_fun()[4])
    plot_training(ID, env_eval.task_fun()[4])
    agent = SAC.load("agent/trained/tmp/best_model.zip", env=env_eval)
    agent.ID = ID
    agent.save(f'agent/trained/{env_eval.task_fun()[4]}_{agent.ID}.zip')
    env_eval = env_type(evaluation=True, task=task)
    env_eval.render(ext_agent=agent)
    return


def run_preexisting(task: Task = AltitudeTask, env_type=CitationNormal, during_training=False):
    env_eval = env_type(evaluation=True, task=task)

    if during_training:
        agent = SAC.load(f"agent/trained/tmp/best_model.zip", env=env_eval)
        env_eval.render(ext_agent=agent)
    else:
        env_eval.render()


def keyboardInterruptHandler(signal, frame):
    print('')
    print('Early stopping. Getting last results...')
    run_preexisting(task=current_task, env_type=env, during_training=True, )
    exit(0)


signal.signal(signal.SIGINT, keyboardInterruptHandler)

########################################################################################################################
# ***** CHOOSE FLIGHT SETTINGS ****** #

# current_task = AltitudeTask
current_task = AttitudeTask

env = CitationNormal
# env = CitationElevRange
# env = CitationAileronEff
# env = CitationRudderStuck
# env = CitationHorzTail
# env = CitationVertTail
# env = CitationIcing
# env = CitationCgShift

# learn(current_task, env)
run_preexisting(current_task, env)

# os.system('say "your program has finished"')
