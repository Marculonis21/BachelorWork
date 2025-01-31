#!/usr/bin/env python
"""GUI Agent tab

Agent tab is the third GUI tab used for selecting and configuring genetic agent
(fully described in the text of the project, see also :mod:`gaAgents`). User
can select any agent available from the selection menu. Selected agent will
display a small overview as well as its parameters in the bottom half of the
tab. These parameters can be altered to change agent behaviour.
"""

import PySimpleGUI as sg
import inspect
from types import FunctionType

import resources.GUI_tabs.evo_tab as evo_tab

import roboEvo
from typing import Dict

DEFAULT_FONT = ("Arial", 15)

agents : Dict[str, roboEvo.gaAgents.BaseAgent]
"""
Dictionary of all available agents by name. 

.. warning::
    User must include new custom agents inside this dictionary to make them
    available through GUI. The source code gives an example how to add new
    entries.

"""

# default for gui init
agents = {agent.__class__.__name__ : agent for agent in [
    roboEvo.gaAgents.StepCycleFullAgent(roboEvo.robots.AntV3(),    [], roboEvo.gaAgents.EvoType.CONTROL, gui=True),
    roboEvo.gaAgents.StepCycleHalfAgent(roboEvo.robots.AntV3(),    [], roboEvo.gaAgents.EvoType.CONTROL, gui=True),
    roboEvo.gaAgents.SineFuncFullAgent(roboEvo.robots.AntV3(),     [], roboEvo.gaAgents.EvoType.CONTROL, gui=True),
    roboEvo.gaAgents.SineFuncHalfAgent(roboEvo.robots.AntV3(),     [], roboEvo.gaAgents.EvoType.CONTROL, gui=True),
    roboEvo.gaAgents.TFSAgent(roboEvo.robots.AntV3(),              [], roboEvo.gaAgents.EvoType.CONTROL, gui=True),
    roboEvo.gaAgents.NEATAgent(roboEvo.robots.AntV3(),             [], roboEvo.gaAgents.EvoType.CONTROL, gui=True)
]}

agent_names = list(agents.keys())

def single_value_option(key, text, tooltip, default, disabled=False):
    text = [sg.Text(text, tooltip=tooltip)]
    input = [sg.Input(default, size=(8,None), enable_events=True, key=key, disabled=disabled)]

    return text, input

def range_value_option(key, text, tooltip, default_min, default_max):
    text = [sg.Text(text, tooltip=tooltip)]
    input = [sg.Text("MIN", font=(DEFAULT_FONT[0], 10)), sg.Input(default_min, size=(8,None), enable_events=True, key=key+"_min"),
             sg.Text("MAX", font=(DEFAULT_FONT[0], 10)), sg.Input(default_max, size=(8,None), enable_events=True, key=key+"_max")]

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

def reset_agent_arguments(agent_name, window):
    agent = agents[agent_name]
    for arg in agent.arguments:
        if isinstance(agent.arguments[arg], dict):
            window[f"{agent_name}|{arg}_min"].update(agent.arguments[arg]["MIN"])
            window[f"{agent_name}|{arg}_max"].update(agent.arguments[arg]["MAX"])
        else:
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
"""
Dictionary of agent arguments corresponding to each agent. Used when selecting
different agent from the selection menu to show the correct arguments for
current agent and to hide all other ones.

This dictionary is filled automatically from the :attr:`agents` dictionary.
"""

agents_gen_ops_default = {agent : get_agent_gen_ops(agent) for agent in agent_names}

def reload_agents(window, robot, agent):
    # reload all agents with current body part settings and reset current agent arguments if agent is set
    for name in agent_names:
        agents[name] = agents[name].__class__(robot, 
                                              [False]*len(robot.body_parts) if agent == None else agent.orig_body_part_mask, 
                                              evo_tab.evo_types[evo_tab.evo_type_names[0]] if agent == None else agent.evo_type, 
                                              gui=True)

        if agent.__class__.__name__ == name:
            if not isinstance(agent, roboEvo.gaAgents.NEATAgent):
                evo_tab.set_evo_ops(name, window)
            reset_agent_arguments(name, window)

    reset_agent_arguments("NEATAgent", window)


def tab():
    default_name = agent_names[0]
    options_menu = [sg.Text("Select agent type: ", pad=(10,None)), sg.Combo(agent_names, default_name, pad=(0,20,0,20), readonly=True, enable_events=True, key="-AGENT_SELECT-")]

    frame = [sg.Frame("Agent overview", [[sg.Text(agents[default_name].description, size=(58, 6), pad=(10,10), key="-AGENT_OVERVIEW-")],
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
    frame = sg.Frame("Description", [[sg.Text(text, size=(60,None), pad=(10,10))]])
    sg.Window("Description long", [[frame]], font=DEFAULT_FONT, keep_on_top=True, modal=True).read(close=True)

def handle_argument_inputs(window, values, key):
    out = ""
    dot = False
    for s in values[key]:
        if s == "-" and out == "":
            out += "-"
        if s in '1234567890':
            out += s
        if s == '.' and not dot:
            dot = True
            out += s
    window[key].update(out)
    values[key] = out

def handle_range_inputs(window, values, key):
    main_key = key[:-4]
    edit = key[-4:]
    if edit == "_min":
        min = values[f"{main_key}_min"]
        max = values[f"{main_key}_max"]
        min = 0 if min in ["", "-", ".", "-."] else float(min)
        max = 0 if max in ["", "-", ".", "-."] else float(max)

        if min > max:
            window[f"{main_key}_max"].update(min)

    if edit == "_max":
        min = values[f"{main_key}_min"]
        max = values[f"{main_key}_max"]
        min = 0 if min in ["", "-", ".", "-."] else float(min)
        max = 0 if max in ["", "-", ".", "-."] else float(max)

        if max < min:
            window[f"{main_key}_min"].update(max)

def events(window, event, values):
    # '|' splits agent arguments
    if '|' in event: 
        handle_argument_inputs(window, values, event)
        if "_min" in event or "_max" in event:
            handle_range_inputs(window, values, event)

    if event == "-AGENT_SELECT-":
        set_agent(values["-AGENT_SELECT-"], window)
        window['-AGENT_SELECT-'].widget.select_clear()
        window.refresh()

    if event == "-AGENT_OVERVIEW_MORE-": 
        expand_description(agents[values['-AGENT_SELECT-']].description)
