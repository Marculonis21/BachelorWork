#!/usr/bin/env python

import PySimpleGUI as sg
import inspect
from types import FunctionType

import resources.GUI_tabs.evo_tab as evo_tab

import roboEvo
from typing import Dict

font = ("Helvetica", 15)

agents : Dict[str, roboEvo.gaAgents.BaseAgent]
# default for gui init
agents = {agent.__class__.__name__ : agent for agent in [
    roboEvo.gaAgents.FullRandomAgent(roboEvo.robots.AntV3(),    [], roboEvo.gaAgents.EvoType.CONTROL, gui=True),
    roboEvo.gaAgents.SineFuncFullAgent(roboEvo.robots.AntV3(),  [], roboEvo.gaAgents.EvoType.CONTROL, gui=True),
    roboEvo.gaAgents.SineFuncHalfAgent(roboEvo.robots.AntV3(),  [], roboEvo.gaAgents.EvoType.CONTROL, gui=True),
    roboEvo.gaAgents.StepCycleHalfAgent(roboEvo.robots.AntV3(), [], roboEvo.gaAgents.EvoType.CONTROL, gui=True),
    roboEvo.gaAgents.TFSAgent(roboEvo.robots.AntV3(),           [], roboEvo.gaAgents.EvoType.CONTROL, gui=True),
    roboEvo.gaAgents.NEATAgent(roboEvo.robots.AntV3(),          [], roboEvo.gaAgents.EvoType.CONTROL, gui=True)
]}

agent_names = list(agents.keys())

def single_value_option(key, text, tooltip, default, disabled=False):
    text = [sg.Text(text, tooltip=tooltip)]
    input = [sg.Input(default, size=(8,None), enable_events=True, key=key, disabled=disabled)]

    return text, input

def range_value_option(key, text, tooltip, default_min, default_max):
    text = [sg.Text(text, tooltip=tooltip)]
    FONT = ("Helvetica", 12)
    input = [sg.Text("MIN", font=FONT), sg.Input(default_min, size=(8,None), enable_events=True, key=key+"_min"),
             sg.Text("MAX", font=FONT), sg.Input(default_max, size=(8,None), enable_events=True, key=key+"_max")]

    return text, input

def get_agent_arguments(agent_name):
    agent = agents[agent_name]

    names = []
    inputs = []

    for arg in agent.arguments:
        # RANGE argument
        if isinstance(agent.arguments[arg], dict):
            tooltip = agent.arguments_tooltips[arg] if arg in agent.arguments_tooltips.keys() else None
            _name, _input = range_value_option(f"{agent_name}|{arg}", str(arg).capitalize(), tooltip, agent.arguments[arg]["MIN"], agent.arguments[arg]["MAX"])
            names.append(_name)
            inputs.append(_input)

        # SINGLE VALUE argument
        else: 
            tooltip = agent.arguments_tooltips[arg] if arg in agent.arguments_tooltips.keys() else None
            _name, _input = single_value_option(f"{agent_name}|{arg}", str(arg).capitalize(), tooltip, agent.arguments[arg], "NET_NUM" in arg)
            names.append(_name)
            inputs.append(_input)

    return [[sg.Column(names), sg.Column(inputs)]]

def reset_agent_arguments(agent_name, window, values):
    agent = agents[agent_name]
    for arg in agent.arguments:
        if isinstance(agent.arguments[arg], dict):
            window[f"{agent_name}|{arg}_min"].update(agent.arguments[arg]["MIN"])
            window[f"{agent_name}|{arg}_max"].update(agent.arguments[arg]["MAX"])

            window[f"{agent_name}|{arg}"].update(agent.arguments[arg])

def get_agent_gen_ops(agent_name):
    agent = agents[agent_name]
    if isinstance(agent, roboEvo.gaAgents.NEATAgent): return None

    def extract_wrapped(decorated):
        closure = (c.cell_contents for c in decorated.__closure__)
        return next((c for c in closure if isinstance(c, FunctionType)), None)

    def get_operator(func):
        source = inspect.getsource(extract_wrapped(func)).split("\n")
        clean_source = [s.strip() for s in source if not s.strip().startswith("#") and s.find("Operators.") > 0]
        op_line = clean_source[0]
        op_name = op_line[op_line.find("Operators.")+10:op_line.find("(")]#)
        return op_name

    selection = get_operator(agent.selection)
    crossover = get_operator(agent.crossover)
    mutation = get_operator(agent.mutation)

    return {"selection":selection,
            "crossover":crossover,
            "mutation": mutation}

agents_argument_options = {agent : sg.Column(get_agent_arguments(agent), pad=(130,None), scrollable=len(get_agent_arguments(agent)[0][0].Rows) > 7, vertical_scroll_only=True, expand_x=True, element_justification='c', key=f"options_{agent}") for agent in agent_names}
agents_gen_ops_default = {agent : get_agent_gen_ops(agent) for agent in agent_names}

def reload_agents(window, values, robot, agent):
    # reload all agents with current body part settings and reset current agent arguments if agent is set
    for name in agent_names:
        agents[name] = agents[name].__class__(robot, 
                                              [False]*len(robot.body_parts) if agent == None else agent.orig_body_part_mask, 
                                              evo_tab.evo_types[evo_tab.evo_type_names[0]] if agent == None else agent.evo_type, 
                                              gui=True)

        if agent.__class__.__name__ == name:
            reset_agent_arguments(name, window, values)
            if not isinstance(agent, roboEvo.gaAgents.NEATAgent):
                evo_tab.set_evo_ops(name, window)
        reset_agent_arguments("NEATAgent", window, values)


def tab():
    default_name = agent_names[0]
    options_menu = [sg.Text("Select agent type: ", pad=(10,None)), sg.Combo(agent_names, default_name, pad=(0,20,0,20), readonly=True, enable_events=True, key="-AGENT_SELECT-")]

    frame = [sg.Frame("Agent overview", [[sg.Text(agents[default_name].description, font=("Helvetica", 14), size=(58, 6), pad=(10,10), key="-AGENT_OVERVIEW-")],
                                         [sg.Push(), sg.Button("...", button_color=sg.theme_background_color(), border_width=0, key="-AGENT_OVERVIEW_MORE-")]], expand_x=True, pad=(10,0))]

    agent_options = []
    for name in agent_names:
        agent_options.append(agents_argument_options[name])

    layout = [options_menu, 
              frame,
              [sg.VPush()],
              agent_options,
              [sg.VPush()]]

    tab = sg.Tab("Agent config", layout, key="-AGENT_TAB-")
    return tab

def set_agent(agent_selected, window):
    agent = agents[agent_selected]

    TEXT = agent.description
    cutoff = 260
    if len(TEXT) > cutoff:
        TEXT = TEXT[:cutoff] + "..."

    window["-AGENT_OVERVIEW_MORE-"].update(visible=len(TEXT)>cutoff)
    window["-AGENT_OVERVIEW-"].update(TEXT)

    # Hide other options 
    for name in agent_names:
        window["options_"+name].update(visible=False)

    # Show selected option
    window["options_"+agent_selected].update(visible=True)

    if not isinstance(agent, roboEvo.gaAgents.NEATAgent):
        evo_tab.set_evo_ops(agent_selected, window)
        window["-EVO_TAB-"].update(visible=True)
    else:
        window["-EVO_TAB-"].update(visible=False)


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
