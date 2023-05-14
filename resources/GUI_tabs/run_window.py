#!/usr/bin/env python

import roboEvo

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import PySimpleGUI as sg

font = ("Helvetica", 15)

figure = None
figure_aggregate = None

def window():
    layout = [[sg.Canvas(key='-FIG-')],
              [sg.Text("Gen Number: "), sg.Text("", key="-GENNUM-")],
              [sg.Text("Max Fit: "), sg.Text("", key="-MAXFIT-")], 
              [sg.Text("Mean Fit: "), sg.Text("", key="-MEANFIT-")], 
              [sg.Text("Min Fit: "), sg.Text("", key="-MINFIT-"), sg.Push(), sg.Button("Preview best", key="-RUN_PREVIEW-"), sg.Button("Exit", button_color='red', mouseover_colors='orange red', key="-EXIT-")]]

    window = sg.Window('Process running', layout, font=font, finalize=True, keep_on_top=True)

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
    plt.plot(roboEvo.GRAPH_VALUES[0], label='Mean')
    plt.plot(roboEvo.GRAPH_VALUES[1], label='Min')
    plt.plot(roboEvo.GRAPH_VALUES[2], label='Max')
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
    plt.plot(roboEvo.GRAPH_VALUES[0], label='Mean')
    plt.plot(roboEvo.GRAPH_VALUES[1], label='Min')
    plt.plot(roboEvo.GRAPH_VALUES[2], label='Max')
    plt.legend(loc='upper left', fontsize=9)
    plt.tight_layout()

    figure_aggregate = draw_figure(window['-FIG-'].TKCanvas, figure)

def get_params(values, robot_tab, agent_tab):
    robot = robot_tab.robots[values["-ROBOT_SELECT-"]] 
    agent = agent_tab.agents[values["-AGENT_SELECT-"]]

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

    agent.arguments = agent_arguments

    population_size = int(values["-MAIN_POP_SIZE_IN-"])
    generation_count = int(values["-MAIN_GEN_COUNT_IN-"])

    show_best = values["-CB_SHOWBEST-"]
    save_best = values["-CB_SAVEBEST-"]
    save_dir = values["Browse"]
    params = roboEvo.ExperimentParams(robot, 
                                      agent, 
                                      population_size, 
                                      generation_count, 
                                      show_best, 
                                      save_best, 
                                      save_dir, 
                                      note="")

    return params
