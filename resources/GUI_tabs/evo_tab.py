#!/usr/bin/env python
"""Evolution Config tab

Evolution config is the last tab. In this tab the user can see and configure
parameters which are used during the evolutionary algorithm run. User can
select the type of evolution they want to run as well as initial size of the
population for evolutionary algorithm and wanted amount of generations for the
algorithm (see also :mod:`gaAgents`).

Then there are three frames for the three genetic operators - Selection,
Crossover and Mutation. In these parts of the tab, user can configure which
method will be used as the genetic operator (automatic operator finding in
:mod:`gaOperators`), as well as all parameters of those operators.
"""

import PySimpleGUI as sg

import roboEvo

import resources.GUI_tabs.agent_tab as agent_tab
import resources.GUI_tabs.robot_tab as robot_tab 

evo_types = {e_type.name : e_type for e_type in roboEvo.gaAgents.EvoType if e_type.name != "BODY"}
evo_type_names = list(evo_types.keys())

ga_operators = roboEvo.gaAgents.Operators._ops_dir()
"""
Dictionary of type ``[operator type, dict[operator name, (operator function, operator args*)]]``. 

It contains all recognised implemented genetic operators inside the
:class:`gaOperators.Operators`, which were collected by the
:func:`gaOperators.Operators._ops_dir`.
"""

DEFAULT_FONT = ("Arial", 15)

def single_value_option(key, text, tooltip, default):
    text = [sg.Text(text, tooltip=tooltip)]
    input = [sg.Input(default, size=(8,None), enable_events=True, key=key)]

    return text, input

def create_op_frame(name, key):
    op_types = list(ga_operators[key].keys())

    combo_type = [sg.Text(f"{name} operator: ", pad=(10,None)), sg.Combo(op_types, op_types[0], size=(25,None), readonly=True, enable_events=True, key=f"-OP_{name.upper()}_TYPE-")]

    options = []

    for op_type in op_types:
        _, op_args, op_args_defaults = ga_operators[key][op_type]
        arg_names = []
        arg_inputs = []

        if op_args != []:
            for op_arg, op_arg_default in zip(op_args, op_args_defaults):
                _name, _input = single_value_option(f"{key}|{op_type}|{op_arg}", str(op_arg).capitalize(), None, op_arg_default)
                arg_names.append(_name)
                arg_inputs.append(_input)

        options.append(sg.Column([[sg.Column(arg_names), sg.Column(arg_inputs)]], expand_x=True, element_justification='c', key=f"options_{key}|{op_type}"))

    return combo_type, options

def tab():
    evo_type = [sg.Text("Select evolution type: ", pad=(10,None)), sg.Combo(evo_type_names, evo_type_names[0], size=(25,None), pad=(0,20,0,20), readonly=True, enable_events=True, key="-EVO_TYPE_SELECT-")]

    sizes = [sg.Text("Generation count: ", pad=(10,None)), sg.Input("150", size=(5,None), enable_events=True, key="-GEN_COUNT-"),
             sg.Text("Starting population: "),             sg.Input("100", size=(5,None), enable_events=True, key="-POP_SIZE-")]

    selection_type, selection_inputs = create_op_frame("Selection", "selection")
    selection = [sg.Frame("Selection", [selection_type, [*selection_inputs]], pad=(10,10), expand_x=True)]

    crossover_type, crossover_inputs = create_op_frame("Crossover", "crossover")
    crossover = [sg.Frame("Crossover ", [crossover_type, [*crossover_inputs]], pad=(10,10), expand_x=True)]

    mutation_type, mutation_inputs = create_op_frame("Mutation", "mutation")
    mutation_probs = [[sg.Text("Mutation probabilities: ", pad=(10,None))],
                      [sg.Text("Individual: ", pad=(10, None)), sg.Input("0.75", size=(6,None), enable_events=True, key="-INDIV_MUT_PROB-"),
                       sg.Text("Action mutation: "),            sg.Input("0.1",  size=(6,None), enable_events=True, key="-ACT_MUT_PROB-"),
                       sg.Text("Body mutation: "),              sg.Input("0.1",  size=(6,None), enable_events=True, key="-BODY_MUT_PROB-")]]

    mutation = [sg.Frame("Mutation", [mutation_type, *mutation_probs, [*mutation_inputs]], pad=(10,10), expand_x=True)]

    layout = [evo_type,
              sizes,
              selection,
              crossover,
              mutation]

    tab = sg.Tab("Evolution config", layout, key="-EVO_TAB-")
    return tab;

def set_evo_ops(agent_selected, window):
    agent_gen_ops = agent_tab.agents_gen_ops_default[agent_selected]

    window["-OP_SELECTION_TYPE-"].update(value=agent_gen_ops["selection"])
    set_operator(agent_gen_ops["selection"], window)
    window["-OP_CROSSOVER_TYPE-"].update(value=agent_gen_ops["crossover"])
    set_operator(agent_gen_ops["crossover"], window)
    window["-OP_MUTATION_TYPE-"].update(value=agent_gen_ops["mutation"])
    set_operator(agent_gen_ops["mutation"], window)

    window["-INDIV_MUT_PROB-"].update(agent_tab.agents[agent_selected].individual_mutation_prob)
    window["-ACT_MUT_PROB-"].update(agent_tab.agents[agent_selected].action_mutation_prob)
    window["-BODY_MUT_PROB-"].update(agent_tab.agents[agent_selected].body_mutation_prob)

def handle_argument_inputs(window, values, key):
    out = ""
    dot = False
    for s in values[key]:
        if s in '1234567890':
            out += s
        if s == '.' and not dot:
            dot = True
            out += s
    window[key].update(out)

def set_operator(operator_selected, window, operator_params=[]):
    op_type = ""
    if "selection" in operator_selected.lower():
        op_type = "selection"
    if "crossover" in operator_selected.lower():
        op_type = "crossover"
    if "mutation" in operator_selected.lower():
        op_type = "mutation"

    for op in ga_operators[op_type].keys():
        key = f"options_{op_type}|{op}"
        if key in window.AllKeysDict.keys():
            window[key].update(visible=False)

    wanted_key = f"options_{op_type}|{operator_selected}"
    if wanted_key in window.AllKeysDict.keys():
        window[wanted_key].update(visible=True)

    if operator_params != []:
        # zipping together operator parameter names from list of all operators
        # and actuall parameter values that we get from outside
        for name, value in zip(ga_operators[op_type][operator_selected][1], operator_params):
            key = f"{op_type}|{operator_selected}|{name}"
            window[key].update(value)


def events(window, event, values):
    if event == "-EVO_TYPE_SELECT-":
        window[event].widget.select_clear()
        agent = agent_tab.agents[values["-AGENT_SELECT-"]]
        agent.evo_type = evo_types[values[event]]
        robot = robot_tab.robots[values["-ROBOT_SELECT-"]]
        agent_tab.reload_agents(window, robot, agent)
        window.refresh()

    if event.startswith("-OP_"):
        set_operator(values[event], window)
        window[event].widget.select_clear()
        window.refresh()

    if event.count("|") == 2 or \
       "-POP_SIZE-" == event or \
       "-GEN_COUNT-" == event or \
       "-INDIV_MUT_PROB-" == event or \
       "-ACT_MUT_PROB-" == event or \
       "-BODY_MUT_PROB-" == event:
        handle_argument_inputs(window,values,event)
