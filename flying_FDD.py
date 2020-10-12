import warnings
import signal

import pandas as pd
from agent.sac import SAC
from agent.policy import LnMlpPolicy
from agent.callback import SaveOnBestReturn
from envs.citation import Citation

from tools.schedule import schedule_kink
from tools.identifier import get_ID
from tools.plot_training import plot_training
from tools.plot_response import get_response
from tools.get_task import get_task_tr_fail

warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')

# todo:  check aileron failure after training with correct observations

# failure_inputs = ['de', 20.05, 3.0]
# failure_inputs = ['da', 1.0, 0.3]
failure_inputs = ['dr', 0.0, -15.0]


# failure_inputs = ['cg', 1.0, 1.04]
# failure_inputs = ['ice', 1.0, 1.9]
# failure_inputs = ['ht', 1.0, 0.1]
# failure_inputs = ['vt', 1.0, 0.0]


# failure_inputs = ['vt', 1.0, 0.0]


def run_preexisting(ID1: str, ID2: str):
    env_eval = Citation(evaluation=True, failure=failure_inputs, FDD=True)

    agents = (SAC.load(f"agent/trained/{get_task_tr_fail()[4]}_{ID1}.zip"),
              SAC.load(f"agent/trained/{get_task_tr_fail()[4]}_{ID2}.zip"))
    get_response(env_eval, agent=agents, ID='FDD_' + ID2, failure=True)


# learn()
run_preexisting('9VZ5VE', '5A50AG_dr')  # general, robust
# run_preexisting('9VZ5VE', '9VZ5VE')  # general, robust

# os.system('say "your program has finished"')
