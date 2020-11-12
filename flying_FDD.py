import warnings
from agent.sac import SAC

from tools.get_task import AltitudeTask, AttitudeTask, BodyRateTask
from envs.h_controller import AltController

warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')


# task = AttitudeTask
task = AltitudeTask


from envs.citation import CitationElevRange
from envs.citation import CitationAileronEff
from envs.citation import CitationRudderStuck
from envs.citation import CitationHorzTail
from envs.citation import CitationVertTail
from envs.citation import CitationIcing
from envs.citation import CitationCgShift


def run_preexisting(ID1: str, ID2: str, env):
    # env_eval = Citation(evaluation=True, FDD=True, task=task)
    env_eval = AltController(evaluation=True, FDD=True, inner_controller=env)

    agents = (SAC.load(f"agent/trained/{env_eval.task_fun()[4]}_{ID1}.zip", env=env_eval),
              SAC.load(f"agent/trained/{env_eval.task_fun()[4]}_{ID2}.zip", env=env_eval))
    agents[1].ID = ID2 + f'_{env_eval.InnerController.failure_input[0]}'
    env_eval.render(agent=agents)


# run_preexisting('P7V00G', 'P7V00G')
# run_preexisting('P7V00G', 'GGFC9G_da')
# run_preexisting('P7V00G', '2DPKKS_de')
run_preexisting('XQ2G4Q_normal', 'XQ2G4Q_normal', CitationElevRange)
run_preexisting('XQ2G4Q_normal', 'XQ2G4Q_normal', CitationAileronEff)
run_preexisting('XQ2G4Q_normal', 'XQ2G4Q_normal', CitationRudderStuck)
run_preexisting('XQ2G4Q_normal', 'XQ2G4Q_normal', CitationHorzTail)
run_preexisting('XQ2G4Q_normal', 'XQ2G4Q_normal', CitationVertTail)
run_preexisting('XQ2G4Q_normal', 'XQ2G4Q_normal', CitationIcing)
run_preexisting('XQ2G4Q_normal', 'XQ2G4Q_normal', CitationCgShift)

# os.system('say "your program has finished"')
