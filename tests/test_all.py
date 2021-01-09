import os
import warnings
import PySimpleGUI as sg


def GUI():
    section1 = [[sg.T('Initial Flight Conditions :')],
                [sg.Text('Initial Altitude [m]:'),
                 sg.InputCombo(values=('2000', '5000'), auto_size_text=True, default_value='2000', key='init_alt')],
                [sg.Text('Initial Speed [m/s]:'),
                 sg.InputCombo(values=('90', '140'), auto_size_text=True, default_value='90', key='init_speed')]]

    section2 = [[sg.T('Failure Type')],
                [sg.InputCombo(values=(
                    'rudder stuck at -15deg', '-70% aileron effectiveness', 'elevator range reduced to [-2.5deg, 2.5deg]',
                    'partial horizontal tail loss', 'partial vertical tail loss', 'c.g. shift', 'severe icing'),
                    default_value='rudder stuck at -15deg', auto_size_text=True, key='fail_type')],
                [sg.Text('Initial altitude and speed are set as 2000m and 90 m/s, respectively.')]]

    section3 = [[sg.T('Controller Structure :')],
                [sg.Radio('Cascaded', 'struct', size=(16, 1), default=True, enable_events=True,
                          key='-OPEN CASC'),
                 sg.Radio('Single', 'struct', size=(16, 1), enable_events=True,
                          key='-OPEN SINGLE')], ]

    fname = 'assets/citation_550.png'

    layout = [[sg.Text('Fault Tolerant Flight Control for the Cessna Citation 500', font=('Helvetica', 18))],
              [sg.Text('with Soft Actor Critic Deep Reinforcement Learning', font=('Helvetica', 18))],
              [sg.Text('Author: Killian Dally, TU Delft (2020)')],
              [sg.Image(filename=fname, size=(440, 140), tooltip='PH-LAB Aircraft')],

              [sg.Text('_' * 100, size=(75, 1))],
              [sg.Text('Flight conditions', font=('Helvetica', 14))],
              [sg.Radio('Normal system', 'condition', size=(15, 1), enable_events=True, key='-OPEN COND-NORM', default=True),
               sg.Radio('Failed system', 'condition', size=(14, 1), enable_events=True, key='-OPEN COND-FAIL')],
              [sg.pin(sg.Column(section1, key='-COND-NORM', visible=True))],
              [sg.pin(sg.Column(section2, key='-COND-FAIL', visible=False))],
              [sg.Checkbox('Sensor noise', default=False, key='sens_noise'),
               sg.Checkbox('Atmospheric disturbances', default=False, key='dist')],

              [sg.Text('_' * 100, size=(75, 1))],
              [sg.Text('Controller Type', font=('Helvetica', 16))],
              [sg.Radio('Altitude tracking ', 'task', size=(16, 1), enable_events=True, key='-OPEN STRUCT-ALT',
                        default=True),
               sg.Radio('Attitude tracking', 'task', size=(16, 1), enable_events=True, key='-OPEN STRUCT-ATT')],
              [sg.pin(sg.Column(section3, key='-STRUCT', visible=True))],
              #
              # [sg.Text('_' * 100, size=(75, 1))],
              # [sg.Text('Other Settings', font=('Helvetica', 14))],

              [sg.Cancel('Run Simulation', key='RUN', tooltip='Sim. time: 20s'), sg.Cancel('Exit', key='EXIT')]]

    window = sg.Window('Control Interface', layout)

    opened1 = True
    opened2 = False
    opened3 = True

    while True:  # Event Loop
        event, instructions = window.read()

        if event == sg.WIN_CLOSED or event == 'RUN' or event == 'EXIT':
            break

        if event.startswith('-OPEN COND-'):
            opened1 = not opened1
            opened2 = not opened2
            window['-OPEN COND-NORM'].update(not opened2 and opened1)
            window['-OPEN COND-FAIL'].update(opened2 and not opened1)
            window['-COND-NORM'].update(visible=opened1)
            window['-COND-FAIL'].update(visible=opened2)

        if event.startswith('-OPEN SINGLE'):
            # window['sens_noise'].update(disabled=True, value=False)
            # window['dist'].update(disabled=True, value=False)
            sg.popup('This mode is not recommended. Unstable response is expected.',title='Warning',
                     custom_text='I understand',background_color='orange red', keep_on_top=True, font=('Helvetica', 13))
            # window['low_pass'].update(disabled=True, value=False)

        if event.startswith('-OPEN STRUCT-ATT') or event.startswith('-OPEN CASC'):
            window['sens_noise'].update(disabled=False)
            window['dist'].update(disabled=False)
            # window['low_pass'].update(disabled=False)

        if event.startswith('-OPEN STRUCT-'):
            opened3 = not opened3
            window['-OPEN STRUCT-ALT'].update(opened3)
            window['-OPEN STRUCT-ATT'].update(not opened3)
            window['-STRUCT'].update(visible=opened3)

    window.close()

    if event == 'EXIT' or event == sg.WIN_CLOSED:
        exit()

    return instructions


def __main__():

    instructions = GUI()

    if not os.path.exists('figures'):
        os.makedirs('figures')

    import fault_tolerant_flight_control_drl as ft

    from tensorflow.python.util import deprecation

    deprecation._PRINT_DEPRECATION_WARNINGS = False
    deprecation._PRINTED_WARNING = False

    warnings.filterwarnings("ignore", module='tensorflow')
    warnings.filterwarnings("ignore", module='gym')

    is_failed = instructions['-OPEN COND-FAIL']
    fail_type = instructions['fail_type']
    init_alt = float(instructions['init_alt'])
    init_speed = float(instructions['init_speed'])

    is_task_alt = instructions['-OPEN STRUCT-ALT']
    is_cascaded = instructions['-OPEN CASC']

    disturbance = instructions['dist']
    sensor_noise = instructions['sens_noise']
    low_pass = False

    if is_failed:
        if fail_type == 'rudder stuck at -15deg':
            env = ft.envs.CitationRudderStuck
        elif fail_type == '-70% aileron effectiveness':
            env = ft.envs.CitationAileronEff
        elif fail_type == 'elevator range reduced to [-2.5deg, 2.5deg]':
            env = ft.envs.CitationElevRange
        elif fail_type == 'partial horizontal tail loss':
            env = ft.envs.CitationHorzTail
        elif fail_type == 'partial vertical tail loss':
            env = ft.envs.CitationVertTail
        elif fail_type == 'c.g. shift':
            env = ft.envs.CitationCgShift
        else:  # fail_type == 'severe icing':
            env = ft.envs.CitationIcing
    else:
        env = ft.envs.CitationNormal

    if is_task_alt:
        if is_cascaded:
            env_eval = ft.envs.AltController(evaluation=True, FDD=is_failed, inner_controller=env,
                                             init_alt=init_alt, init_speed=init_speed, disturbance=disturbance,
                                             sensor_noise=sensor_noise, low_pass=low_pass)
        else:
            env_eval = env(evaluation=True, FDD=is_failed, task=ft.tools.AltitudeTask, init_alt=init_alt,
                           init_speed=init_speed,
                           disturbance=disturbance, sensor_noise=sensor_noise, low_pass=low_pass)
    else:
        env_eval = env(evaluation=True, task=ft.tools.DisturbanceRejectionAtt, FDD=is_failed, init_alt=init_alt, init_speed=init_speed,
                       disturbance=disturbance, sensor_noise=sensor_noise, low_pass=low_pass)

    env_eval.render()


__main__()
