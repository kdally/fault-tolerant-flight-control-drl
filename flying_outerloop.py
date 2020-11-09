import warnings
import signal
import pandas as pd

from agent.sac import SAC
from agent.policy import LnMlpPolicy
from agent.callback import SaveOnBestReturn
from envs.h_controller import AltController
from tools.schedule import schedule_kink, constant
from tools.identifier import get_ID
from tools.plot_training import plot_training
from tools.plot_weights import plot_weights

warnings.filterwarnings("ignore", module='tensorflow')
warnings.filterwarnings("ignore", module='gym')

# from envs.citation import CitationElevRange as Citation
# from envs.citation import CitationAileronEff as Citation
# from envs.citation import CitationRudderStuck as Citation
# from envs.citation import CitationHorzTail as Citation
# from envs.citation import CitationVertTail as Citation
# from envs.citation import CitationIcing as Citation
# from envs.citation import CitationCgShift as Citation
from envs.citation import CitationNormal as Citation


def learn():
    env_train = AltController(inner_controller=Citation)
    env_eval = AltController(inner_controller=Citation)

    callback = SaveOnBestReturn(eval_env=env_eval, eval_freq=2000, log_path="agent/trained/tmp/",
                                best_model_save_path="agent/trained/tmp/")
    agent = SAC(LnMlpPolicy, env_train, verbose=1,
                ent_coef='auto', batch_size=512,
                # learning_rate=schedule_kink(0.0005, 0.0004),
                train_freq=100,
                learning_rate=constant(0.0003),
                policy_kwargs=dict(layers=[32, 32]),
                )
    agent.learn(total_timesteps=int(1.5e6), callback=callback)
    ID = get_ID(6)
    training_log = pd.read_csv('agent/trained/tmp/monitor.csv')
    training_log.to_csv(f'agent/trained/{env_eval.task_fun()[4]}_{ID}.csv')
    plot_weights(ID, env_eval.task_fun()[4])
    plot_training(ID, env_eval.task_fun()[4])
    agent = SAC.load("agent/trained/tmp/best_model.zip", env=env_eval)
    agent.ID = ID + f'_{env_eval.InnerController.failure_input[0]}'
    agent.save(f'agent/trained/{env_eval.task_fun()[4]}_{agent.ID}.zip')
    env_eval = AltController(evaluation=True, inner_controller=Citation)
    env_eval.render(agent=agent)

    return


def run_preexisting(ID=None):
    env_eval = AltController(evaluation=True, inner_controller=Citation)

    if ID is None:
        env_eval.render()
    else:
        agent = SAC.load(f"agent/trained/{env_eval.task_fun()[4]}_{ID}.zip", env=env_eval)
        agent.ID = ID + f'_{env_eval.InnerController.failure_input[0]}'
        env_eval.render(agent=agent)


def keyboardInterruptHandler(signal, frame):
    print('')
    print('Early stopping. Getting last results...')
    run_preexisting()
    exit(0)


signal.signal(signal.SIGINT, keyboardInterruptHandler)
learn()
# run_preexisting('A5FI4R')
# run_preexisting('PZ5QGL')

# os.system('say "your program has finished"')
