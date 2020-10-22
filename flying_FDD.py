import importlib
import warnings
from agent.sac import SAC

from envs.citation import Citation
from tools.get_task import get_task_tr_fail
from tools.identifier import get_ID

warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')

from envs.citation import CitationIcing as Citation


def run_preexisting(ID1: str, ID2: str):
    env_eval = Citation(evaluation=True, FDD=True)

    agents = (SAC.load(f"agent/trained/{get_task_tr_fail()[4]}_{ID1}.zip", env=env_eval),
              SAC.load(f"agent/trained/{get_task_tr_fail()[4]}_{ID2}.zip", env=env_eval))
    agents[1].ID = ID2
    env_eval.render(agent=agents)


# learn()
run_preexisting('9VZ5VE', '9MUWUB_ice')
# run_preexisting('9VZ5VE', '9VZ5VE')  # general, robust

# os.system('say "your program has finished"')
