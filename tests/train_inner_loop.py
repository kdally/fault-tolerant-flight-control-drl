import os
import warnings
import signal
import pandas as pd

import fault_tolerant_flight_control_drl as ft
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
deprecation._PRINTED_WARNING = False

warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')

if not os.path.exists('figures/during_training'):
    os.makedirs('figures/during_training')


def learn(task: ft.tools.Task, env_type=ft.envs.CitationNormal):
    env_train = env_type(task=task)
    env_eval = env_type(task=task)

    callback = ft.agent.SaveOnBestReturn(eval_env=env_eval, log_path="fault_tolerant_flight_control_drl/agent/trained/tmp/",
                                best_model_save_path="fault_tolerant_flight_control_drl/agent/trained/tmp/")
    agent = ft.agent.SAC(ft.agent.LnMlpPolicy, env_train,
                # ent_coef='auto', batch_size=512,
                # learning_rate=schedule_kink(0.0004, 0.0002),
                # train_freq=100,
                # learning_rate=constant(0.0003),
                # policy_kwargs=dict(layers=[32, 32]),
                ent_coef='auto', batch_size=256,
                learning_rate=ft.tools.schedule_kink(0.0004, 0.0002)
                )
    agent.learn(total_timesteps=int(1e6), callback=callback)
    ID = ft.tools.get_ID(6)
    if env_eval.failure_input[0] != 'normal':
        ID += f'_{env_eval.failure_input[0]}'
    training_log = pd.read_csv('fault_tolerant_flight_control_drl/agent/trained/tmp/monitor.csv')
    training_log.to_csv(f'fault_tolerant_flight_control_drl/agent/trained/{env_eval.task_fun()[4]}_{ID}.csv')
    ft.tools.plot_weights(ID, env_eval.task_fun()[4])
    ft.tools.plot_training(ID, env_eval.task_fun()[4])
    agent = ft.agent.SAC.load("fault_tolerant_flight_control_drl/agent/trained/tmp/best_model.zip", env=env_eval)
    agent.ID = ID
    agent.save(f'fault_tolerant_flight_control_drl/agent/trained/{env_eval.task_fun()[4]}_{agent.ID}.zip')
    env_eval = env_type(evaluation=True, task=task)
    env_eval.render(ext_agent=agent)
    return


def run_preexisting(task: ft.tools.Task = ft.tools.AltitudeTask, env_type=ft.envs.CitationNormal, during_training=False):
    env_eval = env_type(evaluation=True, task=task, init_speed=140)

    if during_training:
        agent = ft.agent.SAC.load(f"fault_tolerant_flight_control_drl/agent/trained/tmp/best_model.zip", env=env_eval)
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

current_task = ft.tools.AltitudeTask
# current_task = ft.tools.AttitudeTask

env = ft.envs.CitationNormal
# env = ft.envs.CitationElevRange
# env = ft.envs.CitationAileronEff
# env = ft.envs.CitationRudderStuck
# env = ft.envs.CitationHorzTail
# env = ft.envs.CitationVertTail
# env = ft.envs.CitationIcing
# env = ft.envs.CitationCgShift

learn(current_task, env)
# run_preexisting(current_task, env)

os.system('say "your program has finished"')
