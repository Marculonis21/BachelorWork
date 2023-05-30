#!/usr/bin/env python

import PySimpleGUI as sg

import roboEvo
from typing import Dict

font = ("Helvetica", 15)

agents : Dict[str, roboEvo.gaAgents.BaseAgent]
# default for gui init
agents = {agent.__class__.__name__ : agent for agent in [
    roboEvo.gaAgents.FullRandomAgent(roboEvo.robots.AntV3(), []),
    roboEvo.gaAgents.SineFuncFullAgent(roboEvo.robots.AntV3(), []),
    roboEvo.gaAgents.SineFuncHalfAgent(roboEvo.robots.AntV3(), []),
    roboEvo.gaAgents.StepCycleHalfAgent(roboEvo.robots.AntV3(), []),
    roboEvo.gaAgents.TFSAgent(roboEvo.robots.AntV3(), []),
    roboEvo.gaAgents.NEATAgent(roboEvo.robots.AntV3(), [])
]}

agent_names = list(agents.keys())

# agents_argument_defaults = {}

def single_value_option(key, text, tooltip, default, disabled=False):
    text = [sg.Text(text, tooltip=tooltip)]
    input = [sg.Input(default, size=(8,None), enable_events=True, key=key, disabled=disabled)]

    # global agents_argument_defaults
    # agents_argument_defaults[key] = default

    return text, input

def range_value_option(key, text, tooltip, default_min, default_max):
    text = [sg.Text(text, tooltip=tooltip)]
    FONT = ("Helvetica", 12)
    input = [sg.Text("MIN", font=FONT), sg.Input(default_min, size=(8,None), enable_events=True, key=key+"_min"),
             sg.Text("MAX", font=FONT), sg.Input(default_max, size=(8,None), enable_events=True, key=key+"_max")]

    # global agents_argument_defaults
    # agents_argument_defaults[key+"_min"] = default_min
    # agents_argument_defaults[key+"_max"] = default_max

    return text, input

def get_agent_arguments(agent_name):
    agent = agents[agent_name]

    names = []
    inputs = []

    for arg in agent.arguments:
        # RANGE argument
        if isinstance(agent.arguments[arg], dict):
            tooltip = agent.arguments_tooltips[arg] if arg in list(agent.arguments_tooltips.keys()) else None
            _name, _input = range_value_option(f"{agent_name}|{arg}", str(arg).capitalize(), tooltip, agent.arguments[arg]["MIN"], agent.arguments[arg]["MAX"])
            names.append(_name)
            inputs.append(_input)

        # SINGLE VALUE argument
        else: 
            tooltip = agent.arguments_tooltips[arg] if arg in list(agent.arguments_tooltips.keys()) else None
            _name, _input = single_value_option(f"{agent_name}|{arg}", str(arg).capitalize(), tooltip, agent.arguments[arg], "NET_NUM" in arg or "POP_SIZE" in arg)
            names.append(_name)
            inputs.append(_input)

    return [[sg.Column(names), sg.Column(inputs)]]

def reset_agent_arguments(agent_name, window, values):
    agent = agents[agent_name]
    for arg in agent.arguments:
        if isinstance(agent.arguments[arg], dict):
            window[f"{agent_name}|{arg}_min"].update(agent.arguments[arg]["MIN"])
            window[f"{agent_name}|{arg}_max"].update(agent.arguments[arg]["MAX"])
        else:
            if "POP_SIZE" in arg:
                agent.arguments[arg] = values["-POP_SIZE-"]

            window[f"{agent_name}|{arg}"].update(agent.arguments[arg])

agents_argument_options = {agent : sg.Column(get_agent_arguments(agent), pad=(130,None), scrollable=len(get_agent_arguments(agent)[0][0].Rows) > 7, vertical_scroll_only=True, expand_x=True, element_justification='c', key=f"options_{agent}") for agent in agent_names}

def reload_agents(window, values, robot, agent):
    # reload all agents with current body part settings and reset current agent arguments if agent is set
    for name in agent_names:
        agents[name] = agents[name].__class__(robot, [False]*len(robot.body_parts) if agent == None else agent.orig_body_part_mask)

        if agent.__class__.__name__ == name:
            reset_agent_arguments(name, window, values)
        reset_agent_arguments("NEATAgent", window, values)

def tab():
    agent_names = list(agents.keys())
    default_name = agent_names[0]
    options_menu = [sg.Text("Select agent type: ", pad=(10,None)), sg.Combo(agent_names, default_name, pad=(0,20,0,20), readonly=True, enable_events=True, key="-AGENT_SELECT-")]

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

def handle_argument_inputs(window, values, key):
    out = ""
    for s in values[key]:
        if s in '1234567890.':
            out += s
    window[key].update(out)

def events(window, event, values):
    # '|' splits agent arguments
    if '|' in event: 
        handle_argument_inputs(window, values, event)

    if event == "-AGENT_SELECT-":
        set_agent(values["-AGENT_SELECT-"], window)
        window['-AGENT_SELECT-'].widget.select_clear()
        window.refresh()

    if event == "-AGENT_OVERVIEW_MORE-": 
        expand_description(agents[values['-AGENT_SELECT-']].description)
