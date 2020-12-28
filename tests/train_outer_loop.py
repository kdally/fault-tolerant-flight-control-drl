import os
import warnings
import signal
import pandas as pd

import fault_tolerant_flight_control_drl as ft
from tensorflow.python.util import deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False
deprecation._PRINTED_WARNING = False

warnings.filterwarnings("ignore", module='tensorflow')
warnings.filterwarnings("ignore", module='gym')

if not os.path.exists('figures/during_training'):
    os.makedirs('figures/during_training')


def learn():
    env_train = ft.envs.AltController(inner_controller=ft.envs.CitationNormal)
    env_eval = ft.envs.AltController(inner_controller=ft.envs.CitationNormal)

    callback = ft.agent.SaveOnBestReturn(eval_env=env_eval, eval_freq=2000,
                                         log_path="fault_tolerant_flight_control_drl/agent/trained/tmp/",
                                         best_model_save_path="fault_tolerant_flight_control_drl/agent/trained/tmp/")
    agent = ft.agent.SAC(ft.agent.LnMlpPolicy, env_train,
                         ent_coef='auto', batch_size=512,
                         # learning_rate=schedule_kink(0.0005, 0.0004),
                         train_freq=100,
                         learning_rate=ft.tools.constant(0.0003),
                         policy_kwargs=dict(layers=[32, 32]),
                         )
    agent.learn(total_timesteps=int(1e6), callback=callback)
    ID = ft.tools.get_ID(6)
    if env_eval.InnerController.failure_input[0] != 'normal':
        ID += f'_{env_eval.InnerController.failure_input[0]}'
    training_log = pd.read_csv('fault_tolerant_flight_control_drl/agent/trained/tmp/monitor.csv')
    training_log.to_csv(f'fault_tolerant_flight_control_drl/agent/trained/{env_eval.task_fun()[4]}_{ID}.csv')
    ft.tools.plot_weights(ID, env_eval.task_fun()[4])
    ft.tools.plot_training(ID, env_eval.task_fun()[4])
    agent = ft.agent.SAC.load("fault_tolerant_flight_control_drl/agent/trained/tmp/best_model.zip", env=env_eval)
    agent.ID = ID
    agent.save(f'fault_tolerant_flight_control_drl/agent/trained/{env_eval.task_fun()[4]}_{agent.ID}.zip')
    env_eval = ft.envs.AltController(evaluation=True, inner_controller=ft.envs.CitationNormal)
    env_eval.render(ext_agent=agent)

    return


def run_preexisting(during_training=False):
    env_eval = ft.envs.AltController(evaluation=True, inner_controller=ft.envs.CitationNormal, init_alt=5000,
                                     init_speed=90)

    if during_training:
        agent = ft.agent.SAC.load(f"fault_tolerant_flight_control_drl/agent/trained/tmp/best_model.zip", env=env_eval)
        env_eval.render(ext_agent=agent)
    else:
        env_eval.render()


def keyboardInterruptHandler(signal, frame):
    print('')
    print('Early stopping. Getting last results...')
    run_preexisting()
    exit(0)


signal.signal(signal.SIGINT, keyboardInterruptHandler)

learn()
run_preexisting()

# os.system('say "your program has finished"')
