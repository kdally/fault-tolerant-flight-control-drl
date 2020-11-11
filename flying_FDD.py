import warnings
from agent.sac import SAC

from tools.get_task import AltitudeTask, AttitudeTask, BodyRateTask
from envs.h_controller import AltController

warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')


# task = AttitudeTask
task = AltitudeTask


# from envs.citation import CitationElevRange as Citation
# from envs.citation import CitationAileronEff as Citation # done
# from envs.citation import CitationRudderStuck as Citation
# from envs.citation import CitationHorzTail as Citation
# from envs.citation import CitationVertTail as Citation
from envs.citation import CitationIcing as Citation
# from envs.citation import CitationCgShift as Citation


def run_preexisting(ID1: str, ID2: str):
    # env_eval = Citation(evaluation=True, FDD=True, task=task)
    env_eval = AltController(evaluation=True, FDD=True, inner_controller=Citation)

    agents = (SAC.load(f"agent/trained/{env_eval.task_fun()[4]}_{ID1}.zip", env=env_eval),
              SAC.load(f"agent/trained/{env_eval.task_fun()[4]}_{ID2}.zip", env=env_eval))
    agents[1].ID = ID2 + f'_{env_eval.InnerController.failure_input[0]}'
    env_eval.render(agent=agents)


# run_preexisting('9VZ5VE', '9MUWUB_ice')  # general, robust
# run_preexisting('P7V00G', 'P7V00G')
run_preexisting('A5FI4R', 'A5FI4R')
run_preexisting('PZ5QGL', 'PZ5QGL')
# run_preexisting('P7V00G', 'GGFC9G_da')
# run_preexisting('P7V00G', '2DPKKS_de')

# os.system('say "your program has finished"')
