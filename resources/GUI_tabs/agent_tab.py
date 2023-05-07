#!/usr/bin/env python

import PySimpleGUI as sg

import roboEvo

font = ("Helvetica", 15)

agents : 'dict[str, roboEvo.gaAgents.BaseAgent]'
# TODO: REDO ALL AGENTS WHEN ROBOT CHANGES WITH DEFUALT VALUES - will be much easier to configure
agents = {"Full Random"              : roboEvo.gaAgents.FullRandomAgent.for_GUI(),
          "Sine Function Full"       : roboEvo.gaAgents.SineFuncFullAgent.for_GUI(),
          "Sine Function Half"       : roboEvo.gaAgents.SineFuncHalfAgent.for_GUI(),
          "Step Cycle Half"          : roboEvo.gaAgents.StepCycleHalfAgent.for_GUI(),
          "Truncated Fourier Series" : roboEvo.gaAgents.TFSAgent.for_GUI()}

agent_names = list(agents.keys())

def single_value_option(key, text, default, disabled=False):
        text = [sg.Text(text)]
        input = [sg.Input(default, size=(8,None), enable_events=True, key=key, disabled=disabled)]

        return text, input

def range_value_option(key, text, default_min, default_max, disabled=False):
    text = [sg.Text(text)]
    FONT = ("Helvetica", 12)
    input = [sg.Text("MIN", font=FONT), sg.Input(default_min, size=(8,None), enable_events=True, key=key+"_min", disabled=disabled),
             sg.Text("MAX", font=FONT), sg.Input(default_max, size=(8,None), enable_events=True, key=key+"_max", disabled=disabled)]
    return text, input

def options_FullRandomAgent():
    cycle = single_value_option("cycle_repeat", "Step Count", 25)

    return [[sg.Column([cycle[0]]), sg.Column([cycle[1]])]]

def options_StepCycleHalf():
    cycle = single_value_option("cycle_repeat", "Step Count", 25)
    return [[sg.Column([cycle[0]]), sg.Column([cycle[1]])]]

def options_SineFunctionFull():
    amp_range = range_value_option("amplitude_range", "Amplitude range", 0.5, 4)
    freq_range = range_value_option("frequency_range", "Frequency range", 0.5, 4)

    return [[sg.Column([amp_range[0],freq_range[0]]), sg.Column([amp_range[1],freq_range[1]])]]

def options_SineFunctionHalf():
    amp_range = range_value_option("amplitude_range", "Amplitude range", 0.5, 4)
    freq_range = range_value_option("frequency_range", "Frequency range", 0.5, 4)

    return [[sg.Column([amp_range[0],freq_range[0]]), sg.Column([amp_range[1],freq_range[1]])]]

def options_TruncatedFourierSeries():
    period = single_value_option("period", "Period", 4)
    series_length = single_value_option("series_length", "Truncated series length", 3)
    coef_range = single_value_option("coef_range", "Max coeficient value", 1)

    return [[sg.Column([period[0], series_length[0], coef_range[0]]), sg.Column([period[1], series_length[1], coef_range[1]])]]

agents_argument_options = {"Full Random"              : sg.Column(options_FullRandomAgent()       , expand_x=True, element_justification='c', key="options_Full Random"),
                           "Sine Function Full"       : sg.Column(options_SineFunctionFull()      , expand_x=True, element_justification='c', key="options_Sine Function Full"),
                           "Sine Function Half"       : sg.Column(options_SineFunctionHalf()      , expand_x=True, element_justification='c', key="options_Sine Function Half"),
                           "Step Cycle Half"          : sg.Column(options_StepCycleHalf()         , expand_x=True, element_justification='c', key="options_Step Cycle Half"),
                           "Truncated Fourier Series" : sg.Column(options_TruncatedFourierSeries(), expand_x=True, element_justification='c', key="options_Truncated Fourier Series")}

def tab():
    agent_names = list(agents.keys())
    default_name = agent_names[0]
    options_menu = [sg.Text("Select agent type: "), sg.Combo(agent_names, default_name, pad=(0,20,0,20), readonly=True, enable_events=True, key="-AGENT_SELECT-")]

    frame = [sg.Frame("Agent overview", [[sg.Text(agents[default_name].description, font=("Helvetica", 14), size=(58, 6), pad=(10,10), key="-AGENT_OVERVIEW-")],
                                         [sg.Push(), sg.Button("...", button_color=sg.theme_background_color(), border_width=0, key="-AGENT_OVERVIEW_MORE-")]], expand_x=True, pad=(10,0))]


    agent_options = []
    for name in agent_names:
        agent_options.append(agents_argument_options[name])

    tab = sg.Tab("Agent config", [options_menu, 
                                  frame,
                                  [sg.VPush()],
                                  agent_options,
                                  [sg.VPush()]])
    return tab

def set_agent(agent_selected, window):
    TEXT = agents[agent_selected].description
    cutoff = 260
    if len(TEXT) > cutoff:
        TEXT = TEXT[:cutoff] + "..."

    window["-AGENT_OVERVIEW_MORE-"].update(visible=len(TEXT)>cutoff)
    window["-AGENT_OVERVIEW-"].update(TEXT)

    # Hide other options 
    agent_names = list(agents.keys())
    for name in agent_names:
        window["options_"+name].update(visible=False)

    # Show selected option
    window["options_"+agent_selected].update(visible=True)

def expand_description(text):
    frame = sg.Frame("Description", [[sg.Text(text, size=(60,None), font=("Helvetica", 14), pad=(10,10))]])
    sg.Window("Description long", [[frame]], font=font, keep_on_top=True, modal=True).read(close=True)

def events(window, event, values):
    if event == "-AGENT_SELECT-":
        set_agent(values["-AGENT_SELECT-"], window)
        window['-AGENT_SELECT-'].widget.select_clear()
        window.refresh()

    if event == "-AGENT_OVERVIEW_MORE-": 
        expand_description(agents[values['-AGENT_SELECT-']].description)
