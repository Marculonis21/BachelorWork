#!/usr/bin/env python
"""
Window shown when running the experiment from GUI. It contains a graph which is
actively updated while the experiment runs. It gives us the ability to
preview the current best individual and to abort the experiment. When the
experiment finishes, the ``Exit`` button turns green to show that the application
has finished all after experiment procedures.
"""

import roboEvo

import resources.GUI_tabs.evo_tab as evo_tab

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import PySimpleGUI as sg

import numpy as np

DEFAULT_FONT = ("Arial", 15)

figure = None
figure_aggregate = None

def window():
    layout = [[sg.Canvas(key='-FIG-')],
              [sg.Text("Gen Number: "), sg.Text("", key="-GENNUM-")],
              [sg.Text("Max Fit: "), sg.Text("", key="-MAXFIT-")], 
              [sg.Text("Mean Fit: "), sg.Text("", key="-MEANFIT-")], 
              [sg.Text("Min Fit: "), sg.Text("", key="-MINFIT-"), sg.Push(), sg.Button("Preview best", key="-RUN_PREVIEW-"), sg.Button("Exit", button_color='red', mouseover_colors='orange red', key="-EXIT-")]]

    window = sg.Window('Process running', layout, font=DEFAULT_FONT, finalize=True, keep_on_top=True)

    return window

def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw_idle()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

def draw_chart(window):
    global figure, figure_aggregate
    figure = plt.figure()
    plt.title('Training')
    plt.xlabel('Episode')
    plt.ylabel('Fitness')
    if len(roboEvo.EPISODE_HISTORY) > 0:
        plt.plot(np.mean(roboEvo.EPISODE_HISTORY, axis=1), label='Mean')
        plt.plot(np.min( roboEvo.EPISODE_HISTORY, axis=1), label='Min')
        plt.plot(np.max( roboEvo.EPISODE_HISTORY, axis=1), label='Max')
        plt.legend(loc='upper left', fontsize=9)
    plt.tight_layout()
    figure_aggregate = draw_figure(window['-FIG-'].TKCanvas, figure)

def update_chart(window):
    global figure, figure_aggregate
    assert figure_aggregate is not None

    figure_aggregate.get_tk_widget().forget()
    plt.clf()
    plt.title('Training')
    plt.xlabel('Episode')
    plt.ylabel('Fitness')
    if len(roboEvo.EPISODE_HISTORY) > 0:
        plt.plot(np.mean(roboEvo.EPISODE_HISTORY, axis=1), label='Mean')
        plt.plot(np.min( roboEvo.EPISODE_HISTORY, axis=1), label='Min')
        plt.plot(np.max( roboEvo.EPISODE_HISTORY, axis=1), label='Max')
        plt.legend(loc='upper left', fontsize=9)
    plt.tight_layout()

    figure_aggregate = draw_figure(window['-FIG-'].TKCanvas, figure)

def get_params(values, robot_tab, agent_tab):
    # set robot and agent
    robot = robot_tab.robots[values["-ROBOT_SELECT-"]] 
    agent = agent_tab.agents[values["-AGENT_SELECT-"]]

    # get all agent arguments
    agent_arguments = {}
    for key in values:
        if values["-AGENT_SELECT-"] in str(key):
            arg_name = key.split("|").pop()
            arg_value = values[key]

            if arg_name.endswith("_min"):
                arg_name = arg_name.rstrip("_min")
                agent_arguments[arg_name] = {"MIN":arg_value}
            elif arg_name.endswith("_max"):
                arg_name = arg_name.rstrip("_max")
                agent_arguments[arg_name]["MAX"] = arg_value
            else:
                agent_arguments[arg_name] = arg_value

    # Transfer argument options to agent
    for key, value in agent.arguments.items():
        if isinstance(value, dict): # min max range
            if isinstance(value["MIN"], int):
                agent.arguments[key]["MIN"] = int(agent_arguments[key]["MIN"])
            else:
                agent.arguments[key]["MIN"] = float(agent_arguments[key]["MIN"])

            if isinstance(value["MAX"], int):
                agent.arguments[key]["MAX"] = int(agent_arguments[key]["MAX"])
            else:
                agent.arguments[key]["MAX"] = float(agent_arguments[key]["MAX"])
        elif isinstance(value, int):
            agent.arguments[key] = int(agent_arguments[key])
        else:
            agent.arguments[key] = float(agent_arguments[key])

    # get genetic operators selected in evo_tab
    agents_gen_ops = {op_type : (evo_tab.ga_operators[op_type][values[f"-OP_{op_type.upper()}_TYPE-"]][0], []) for op_type in ["selection", "crossover", "mutation"]}
    agents_gen_ops_names = [values[f"-OP_{op_type.upper()}_TYPE-"] for op_type in ["selection", "crossover", "mutation"]]

    for name, op_type in zip(agents_gen_ops_names, list(agents_gen_ops.keys())):
        for key in values.keys():
            if not isinstance(key, int) and name in key:
                agents_gen_ops[op_type][1].append(float(values[key]))

    agent.genetic_operators = agents_gen_ops

    # mutation probabilities from evo_tab
    agent.individual_mutation_prob = float(values["-INDIV_MUT_PROB-"])
    agent.action_mutation_prob     = float(values["-ACT_MUT_PROB-"])
    agent.body_mutation_prob       = float(values["-BODY_MUT_PROB-"])

    population_size = int(values["-POP_SIZE-"])
    generation_count = int(values["-GEN_COUNT-"])

    if isinstance(agent, roboEvo.gaAgents.NEATAgent):
        population_size = int(agent.arguments["POP_SIZE"])
        generation_count = int(agent.arguments["GEN_COUNT"])

    show_best = values["-SHOW_BEST-"]
    save_best = values["-SAVE_BEST-"]
    save_dir = values["Browse"]
    if save_dir == '':
        save_dir = "./saves/individuals/"

    # returning experiment parameters
    params = roboEvo.ExperimentParams(robot, 
                                      agent, 
                                      population_size, 
                                      generation_count, 
                                      show_best, 
                                      save_best, 
                                      save_dir,
                                      note="GUIRun")

    return params
