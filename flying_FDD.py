import warnings
from agent.sac import SAC

from tools.get_task import AltitudeTask, AttitudeTask, BodyRateTask

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
    env_eval = Citation(evaluation=True, FDD=True, task=task)

    agents = (SAC.load(f"agent/trained/{env_eval.task_fun()[4]}_{ID1}.zip", env=env_eval),
              SAC.load(f"agent/trained/{env_eval.task_fun()[4]}_{ID2}.zip", env=env_eval))
    agents[1].ID = ID2
    env_eval.render(agent=agents)


# learn()
run_preexisting('P7V00G', '3QU8VF_ice')
# run_preexisting('9VZ5VE', '9VZ5VE')  # general, robust

# os.system('say "your program has finished"')
