import warnings
from agent.sac import SAC

from tools.get_task import AltitudeTask, AttitudeTask, BodyRateTask, Task, CascadedAltTask
from envs.h_controller import AltController

warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')

from envs.citation import CitationElevRange
from envs.citation import CitationAileronEff
from envs.citation import CitationRudderStuck
from envs.citation import CitationHorzTail
from envs.citation import CitationVertTail
from envs.citation import CitationIcing
from envs.citation import CitationCgShift


def run_preexisting(task: Task, env_type):

    if task == CascadedAltTask:
        env_eval = AltController(evaluation=True, FDD=True, inner_controller=env_type)
    else:
        env_eval = env_type(evaluation=True, FDD=True, task=task)

    env_eval.render()


########################################################################################################################
# ***** CHOOSE FLIGHT SETTINGS ****** #

# env = CitationElevRange
# env = CitationAileronEff
# env = CitationRudderStuck
# env = CitationHorzTail
# env = CitationVertTail
# env = CitationIcing
# env = CitationCgShift

current_task = CascadedAltTask
# current_task = AltitudeTask
# current_task = AttitudeTask

# run_preexisting(current_task, env)

# run_preexisting(current_task, CitationElevRange)
# run_preexisting(current_task, CitationAileronEff)
# run_preexisting(current_task, CitationRudderStuck)
# run_preexisting(current_task, CitationHorzTail)
# run_preexisting(current_task, CitationVertTail)
# run_preexisting(current_task, CitationIcing)
run_preexisting(current_task, CitationCgShift)



# os.system('say "your program has finished"')
