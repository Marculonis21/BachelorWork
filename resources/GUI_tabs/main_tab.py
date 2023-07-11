#!/usr/bin/env python

import PySimpleGUI as sg

import roboEvo
from experiment_setter import Experiments
import resources.GUI_tabs.run_window as run_window 

experiments = Experiments()

FONT = "Arial"

def tab():
    frame_text = [[sg.Text("", font=(FONT, 14), size=(58, None), pad=(10,10), key="-MAIN_SETTINGS_OVERVIEW-")]]

    frame = [sg.Frame("Experiment overview", frame_text, size=(800, 350), pad=(10,10))]
    options = [sg.Checkbox("Save best", visible=False, default=True, key="-SAVE_BEST-"), sg.Checkbox("Show final run", visible=False, key="-SHOW_BEST-"), 
               sg.Button("View individual", pad=((10,0),None), key="-VIEW_INDIVIDUAL-"),
               sg.Push(), 
               sg.Button("Save experiment", pad=((5,0),None), key="-SAVE_EXPERIMENT-"), 
               sg.Button("Load experiment", pad=((2,10),None), key="-LOAD_EXPERIMENT-")]

    save_dir = [sg.Text("Save directory:", pad=((10,0),30)), sg.Text("./saves/individuals/", size=(40,None), font=(FONT, 10), key="-SAVE_DIR_TEXT-"), 
                sg.Push(), 
                sg.FolderBrowse("Browse", size=(6,None), initial_folder="./saves/individuals", pad=((0,10),None), target="-SAVE_DIR_TEXT-")]

    start = [sg.Push(), sg.Button("Start", size=(5,1), pad=(10,5), key="-START-")]

    main = [frame,
            options,
            save_dir,
            [sg.VPush()],
            start]

    tab = sg.Tab("Main", main, key="-MAIN_TAB-")
    return tab

def popup_view_individual():
    title = [sg.Text("Select individual for run view", size=(None,1), pad=(10,10))]

    save_dir = [sg.Text("Selected directory: ", pad=((10,0),30)), sg.Text("", size=(40,None), font=(FONT, 10), key="DIR"), 
                sg.Push(), 
                # sg.FileBrowse("Browse", size=(6,None), initial_folder="./saves/", pad=((0,10),None), file_types=(("Best individual save files","*.save"), ("Population save files" ,"*last_population*")), target="DIR")]
                sg.FileBrowse("Browse", size=(6,None), initial_folder="./saves/", pad=((0,10),None), file_types=(("Best individual save files","*.save")), target="DIR")]

    show = [sg.Button("Show", pad=(10,10), key="SHOW")]
    main = [save_dir,
            show]

    layout = [title,
              main]

    name = None
    popup = sg.Window("Experiment name", layout, font=(FONT, 14), keep_on_top=True, modal=True)
    while True:
        event, values = popup.read(1)

        if event == sg.WIN_CLOSED or event == 'Exit':
            break

        if event == "SHOW" and values["Browse"] != "": # if indiv selected -> play
            agent, robot, individual = roboEvo.gaAgents.BaseAgent.load(values["Browse"])
            run_reward = roboEvo.render_run(agent, robot, individual)
            print("Run reward:", run_reward)

        if popup["DIR"].get() != "":
            popup["DIR"].update(".../"+popup["DIR"].get().split("/").pop()) # makes nicer text

        popup.refresh()
    popup.close()
    return name

def popup_save_experiments():
    title = [sg.Text("Set experiment name for saving:", size=(None,1), pad=(10,10))]

    input = [sg.Input("experiment_name", size=(None,None), pad=(10,None), enable_events=True, key="NAME_INPUT")]
    save_button = [sg.Button("Save", pad=(10,10), key="SAVE")]

    main = [input,
            save_button]

    layout = [title,
              main]

    name = None
    popup = sg.Window("Experiment name", layout, font=(FONT, 14), keep_on_top=True, modal=True)
    while True:
        event, values = popup.read()

        if event == sg.WIN_CLOSED or event == 'Exit':
            break

        if event == "SAVE":
            name = values["NAME_INPUT"]
            break

        popup.refresh()
    popup.close()
    return name

def popup_load_experiments():
    experiments = Experiments()

    title = [sg.Text("Select experiment to be loaded in", size=(None,2), pad=(10,10))]

    main = [[sg.Listbox(experiments.get_experiment_names(), select_mode=sg.SELECT_MODE_SINGLE, font=(FONT, 12), expand_x=True, expand_y=True, pad=(10,5), key="SELECTED_EXPERIMENT")],
            [sg.Button("Load", key="LOAD")]]

    layout = [title,
              main]

    params = None
    popup = sg.Window("Load experiment", layout, size=(700,400), font=(FONT, 14), keep_on_top=True, modal=True)
    while True:
        event, values = popup.read()

        if event == sg.WIN_CLOSED or event == 'Exit':
            break

        if event == "LOAD":
            params = experiments.get_experiment(values["SELECTED_EXPERIMENT"][0])
            break

        popup.refresh()
    popup.close()
    return params

def set_overview_text(window, values, robot_tab, agent_tab):
    _robot = robot_tab.robots[values['-ROBOT_SELECT-']]
    _agent = agent_tab.agents[values['-AGENT_SELECT-']]
    TEXT = "" #'''- Robot selected: {}\n- Robot controlling agent: {}'''.format(values['-ROBOT_SELECT-'], values["-AGENT_SELECT-"])
    TEXT += "- Selected Robot: {}\n".format(values['-ROBOT_SELECT-'])
    for i, part in enumerate(_robot.body_parts):
        if i == 0:
            TEXT += "    - Body parts for GA:\n"
        TEXT += "        - {} ... {}\n".format(part, _agent.orig_body_part_mask[i] if _agent.orig_body_part_mask[i] else "locked")

    TEXT += "\n"

    TEXT += "- Selected Agent: {}\n".format(values['-AGENT_SELECT-'])

    window["-MAIN_SETTINGS_OVERVIEW-"].update(TEXT)

def correct_int_inputs(window, values, key):
    out = ""
    for s in values[key]:
        if s in '1234567890':
            out += s
    window[key].update(out)

def events(window, event, values, robot_tab, agent_tab, evo_tab):
    if event == '-MAIN_GEN_COUNT_IN-': 
        correct_int_inputs(window, values, '-MAIN_GEN_COUNT_IN-')
    if event == '-MAIN_POP_SIZE_IN-':
        correct_int_inputs(window, values, '-MAIN_POP_SIZE_IN-')

    window["-SAVE_EXPERIMENT-"].update("Save experiment")

    if event == "-VIEW_INDIVIDUAL-":
        popup_view_individual()

    if event == "-SAVE_EXPERIMENT-":
        experiment_name = popup_save_experiments()
        if experiment_name != None:
            params = run_window.get_params(values, robot_tab, agent_tab)
            experiments.save_experiment(experiment_name, params)
            window["-SAVE_EXPERIMENT-"].update("SAVED!")


    if event == "-LOAD_EXPERIMENT-":
        experiment_params = popup_load_experiments()

        if experiment_params != None: 
            robot_name = experiment_params.robot.__class__.__name__
            agent_name = experiment_params.agent.__class__.__name__

            window["-ROBOT_SELECT-"].update(robot_name)
            window["-AGENT_SELECT-"].update(agent_name)
            window["-EVO_TYPE_SELECT-"].update(experiment_params.agent.evo_type)
            robot_tab.set_robot(robot_name, window, values, experiment_params.agent)
            agent_tab.set_agent(agent_name, window)
            agent_tab.agents[agent_name] = experiment_params.agent

            if experiment_params.agent.gui:
                selection_f, selection_p = experiment_params.agent.genetic_operators['selection']
                crossover_f, crossover_p = experiment_params.agent.genetic_operators['crossover']
                mutation_f,  mutation_p  = experiment_params.agent.genetic_operators['mutation']

                evo_tab.set_operator(selection_f.__name__, window, selection_p)
                evo_tab.set_operator(crossover_f.__name__, window, crossover_p)
                evo_tab.set_operator(mutation_f.__name__,  window, mutation_p)

            values['-ROBOT_SELECT-'] = robot_name
            values['-AGENT_SELECT-'] = agent_name

            window['-POP_SIZE-'].update(experiment_params.population_size)
            window['-GEN_COUNT-'].update(experiment_params.generation_count)

            window["-INDIV_MUT_PROB-"].update(experiment_params.agent.individual_mutation_prob)
            window["-ACT_MUT_PROB-"].update(experiment_params.agent.action_mutation_prob)
            window["-BODY_MUT_PROB-"].update(experiment_params.agent.body_mutation_prob)

            window['-SAVE_BEST-'].update(experiment_params.save_best)
            window['-SHOW_BEST-'].update(experiment_params.show_best)

            window['-SAVE_DIR_TEXT-'].update(experiment_params.save_dir)

            set_overview_text(window, values, robot_tab, agent_tab)
