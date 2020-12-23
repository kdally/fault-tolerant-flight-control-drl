import warnings
import PySimpleGUI as sg
from tools.get_task import AltitudeTask, AttitudeTask, BodyRateTask, Task, CascadedAltTask
from envs.h_controller import AltController

from tensorflow.python.util import deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False
deprecation._PRINTED_WARNING = False

warnings.filterwarnings("ignore", module='tensorflow')
warnings.filterwarnings("ignore", module='gym')

from envs.citation import CitationElevRange
from envs.citation import CitationAileronEff
from envs.citation import CitationRudderStuck
from envs.citation import CitationHorzTail
from envs.citation import CitationVertTail
from envs.citation import CitationIcing
from envs.citation import CitationCgShift
from envs.citation import CitationNormal


def GUI():
    section1 = [[sg.T('Initial Flight Conditions :')],
                [sg.Text('Initial Altitude [m]:'),
                 sg.InputCombo(values=('2000', '5000'), auto_size_text=True, default_value='2000')],
                [sg.Text('Initial Speed [m/s]:'),
                 sg.InputCombo(values=('90', '140'), auto_size_text=True, default_value='90')]]

    section2 = [[sg.T('Failure Type')],
                [sg.InputCombo(values=(
                'rudder stuck at -15deg', '-70% aileron effectiveness', 'elevator range reduced to [-3deg, 3deg]',
                'partial horizontal tail loss', 'partial vertical tail loss', 'c.g. shift', 'severe icing'),
                               default_value='rudder stuck at -15deg', auto_size_text=True)],
                [sg.Text('Initial conditions are fixed at 2000m and 90 m/s for the altitude and speed, respectively.')]]

    section3 = [[sg.T('Controller Structure :')],
                [sg.Radio('Cascaded (recommended)', 'struct', size=(25, 1), default=True),
                 sg.Radio('Single (not recommended)', 'struct', size=(25, 1))], ]

    layout = [[sg.Text('Fault Tolerant Flight Control for the Cessna Citation 500', font=('Helvetica', 18))],
              [sg.Text('with Soft Actor Critic Deep Reinforcement Learning', font=('Helvetica', 18))],
              [sg.Text('Author: Killian Dally, TU Delft (2020)')],

              [sg.Text('_' * 100, size=(75, 1))],
              [sg.Text('Aircraft condition', font=('Helvetica', 14))],
              [sg.Radio('Normal', 'condition', size=(12, 1), enable_events=True, key='-OPEN COND-NORM', default=True),
               sg.Radio('Failed', 'condition', size=(12, 1), enable_events=True, key='-OPEN COND-FAIL')],
              [sg.pin(sg.Column(section1, key='-COND-NORM', visible=True))],
              [sg.pin(sg.Column(section2, key='-COND-FAIL', visible=False))],

              [sg.Text('_' * 100, size=(75, 1))],
              [sg.Text('Controller Type', font=('Helvetica', 14))],
              [sg.Radio('Altitude tracking ', 'task', size=(16, 1), enable_events=True, key='-OPEN STRUCT-ALT',
                        default=True),
               sg.Radio('Attitude tracking', 'task', size=(16, 1), enable_events=True, key='-OPEN STRUCT-ATT')],
              [sg.pin(sg.Column(section3, key='-STRUCT', visible=True))],

              [sg.Text('_' * 100, size=(75, 1))],
              [sg.Text('Other Settings (not recommended)', font=('Helvetica', 14))],
              [sg.Checkbox('Sensor noise', default=False), sg.Checkbox('Wind disturbance', default=False),
               sg.Checkbox('Low pass filter', default=False)],

              [sg.CloseButton('Run Simulation', key='Run'), sg.Cancel('Exit')]]

    window = sg.Window('Control Interface', layout)

    opened1 = True
    opened2 = False
    opened3 = True

    while True:  # Event Loop
        event, instructions = window.read()
        if event == sg.WIN_CLOSED or event == 'Run' or event == 'Exit':
            break

        if event.startswith('-OPEN COND-'):
            opened1 = not opened1
            opened2 = not opened2
            window['-OPEN COND-NORM'].update(not opened2 and opened1)
            window['-OPEN COND-FAIL'].update(opened2 and not opened1)
            window['-COND-NORM'].update(visible=opened1)
            window['-COND-FAIL'].update(visible=opened2)

        if event.startswith('-OPEN STRUCT-'):
            opened3 = not opened3
            window['-OPEN STRUCT-ALT'].update(opened3)
            window['-OPEN STRUCT-ATT'].update(not opened3)
            window['-STRUCT'].update(visible=opened3)

    window.close()

    if event == 'Exit' or event == sg.WIN_CLOSED:
        exit()

    return instructions


def __main__():

    instructions = GUI()

    is_failed = instructions['-OPEN COND-FAIL']
    fail_type = instructions[3]
    init_alt = float(instructions[0])
    init_speed = float(instructions[1])

    is_task_alt = instructions['-OPEN STRUCT-ALT']
    is_cascaded = instructions[5]

    if is_failed:
        if fail_type == 'rudder stuck at -15deg':
            env = CitationRudderStuck
        elif fail_type == '-70% aileron effectiveness':
            env = CitationAileronEff
        elif fail_type == 'elevator range reduced to [-3deg, 3deg]':
            env = CitationElevRange
        elif fail_type == 'partial horizontal tail loss':
            env = CitationHorzTail
        elif fail_type == 'partial vertical tail loss':
            env = CitationVertTail
        elif fail_type == 'c.g. shift':
            env = CitationCgShift
        else:  # fail_type == 'severe icing':
            env = CitationIcing
    else:
        env = CitationNormal

    if is_task_alt:
        if is_cascaded:
            env_eval = AltController(evaluation=True, FDD=is_failed, inner_controller=env,
                                     init_alt=init_alt, init_speed=init_speed)
        else:
            env_eval = env(evaluation=True, FDD=is_failed, task=AltitudeTask, init_alt=init_alt, init_speed=init_speed)
    else:
        print('here')
        env_eval = env(evaluation=True, task=AttitudeTask, FDD=is_failed, init_alt=init_alt, init_speed=init_speed)

    env_eval.render()


__main__()
