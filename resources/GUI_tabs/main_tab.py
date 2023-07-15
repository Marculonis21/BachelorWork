#!/usr/bin/env python
"""GUI Main tab

This is the first of four tabs that opens with the graphical user interface. On
this tab the user can see small overview of current experiment configuration.
Then there are buttons for saving current configuration and loading previously
created ones. User can also open a popup for viewing individual runs. Lastly,
user can set save directory for the experiment results and start the current
experiment.
"""

import PySimpleGUI as sg

import os
import numpy as np

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

    def load_individuals(file_path, popup):
        """
        A function to find load and evaluate all individuals from selected last 
        generation to be used as items in the listbox.
        """
        parent_dir = os.path.dirname(file_path)

        last_population = np.load(file_path, allow_pickle=True)

        run_timestamp = file_path[file_path.find("population")+10:file_path.find(".npy")]
        best_individual_path = [x for x in os.listdir(parent_dir) if  run_timestamp in x and x.endswith(".save")][0]

        agent, robot, _ = roboEvo.gaAgents.BaseAgent.load(parent_dir+"/"+best_individual_path)

        individual_items = []

        # evaluation
        for index, individual in enumerate(last_population):
            popup["PBAR"].update(current_count=100*index//len(last_population))
            file = robot.create(agent.body_part_mask, individual)
            file.close()
            env = roboEvo.gym.make(id=robot.environment_id,
                                   xml_file=file.name,
                                   reset_noise_scale=0.0,
                                   disable_env_checker=True,
                                   render_mode=None)
            env = roboEvo.TimeLimit(env, max_episode_steps=500)

            run_reward = roboEvo._simulation_run(env, agent, individual)

            individual_items.append((index, run_reward, individual))

        individual_items = sorted(individual_items, key=lambda x: x[1], reverse=True)
        return individual_items, agent, robot

    title = [sg.Text("Select last generation data for run selection", size=(None,1), pad=(10,10))]

    save_dir = [sg.Text("Selected file: ", pad=((10,0),30)), sg.Text("", size=(40,None), font=(FONT, 10), key="DIR"),
                sg.Push(), 
                sg.FileBrowse("Browse", size=(6,None), initial_folder="./saves/", pad=((0,10),None), file_types=((("Last popuulation data","*last_population*")),), target="DIR")]

    progress_bar = [sg.ProgressBar(100, orientation='h', expand_x=True, size=(10,20), key="PBAR", visible=False)]
    indiv_list = [sg.Listbox([], select_mode=sg.SELECT_MODE_SINGLE, font=(FONT, 14), size=(None, 20), expand_x=True, pad=(10,5), key="SELECTION_LISTBOX", disabled=True)]

    show = [sg.Button("Show", pad=(10,10), key="SHOW")]

    main = [save_dir,
            progress_bar,
            indiv_list,
            show]

    layout = [title,
              main]

    agent = None
    robot = None
    sorted_items = []
    last_browse = ""
    popup = sg.Window("Experiment name", layout, font=(FONT, 14), keep_on_top=True, modal=True)
    while True:
        event, values = popup.read(1)

        if event == sg.WIN_CLOSED or event == 'Exit':
            break

        if popup["DIR"].get() != "":
            popup["DIR"].update(".../"+popup["DIR"].get().split("/").pop()) # makes nicer text

        if last_browse != values["Browse"]:
            popup["PBAR"].update(visible=True)
            popup["PBAR"].update(current_count=0)

            sorted_items, agent, robot = load_individuals(values["Browse"], popup)

            item_texts = [f"Individual {x[0]} - Reward: {x[1]}" for x in sorted_items]
            popup["SELECTION_LISTBOX"].update(values=item_texts, disabled=False)

            popup["PBAR"].update(visible=False)
            last_browse = values["Browse"]

        if event == "SHOW" and values["Browse"] != "" and agent != None and robot != None: # if indiv selected -> play
            selected_index = popup["SELECTION_LISTBOX"].Widget.curselection()
            
            # test for no selected item in listbox
            if selected_index == set(): continue 

            individual = sorted_items[selected_index[0]][2]
            run_reward = roboEvo.render_run(agent, robot, individual)
            print("Run reward:", run_reward)

        popup.refresh()
    popup.close()

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
            robot_tab.set_robot(robot_name, window, experiment_params.agent)
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
