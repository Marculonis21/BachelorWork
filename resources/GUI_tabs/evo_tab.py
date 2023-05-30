#!/usr/bin/env python

import PySimpleGUI as sg

import roboEvo

evo_types = [e_type.name for e_type in roboEvo.gaAgents.EvoType if e_type.name != "BODY"]
ga_operators = roboEvo.gaOperators.Operators.__ops_dir__()

FONT = ("Helvetica", 14)
def single_value_option(key, text, tooltip, default):
    text = [sg.Text(text, font=FONT, tooltip=tooltip, key=key+"_text")]
    input = [sg.Input(default, font=FONT, size=(8,None), enable_events=True, key=key)]

    return text, input

def create_op_frame(name, key):
    op_types = list(ga_operators[key].keys())

    combo_type = [sg.Text(f"{name} operator: ", pad=(10,None)), sg.Combo(op_types, op_types[0], font=FONT, size=(25,None), readonly=True, enable_events=True, key=f"-OP_{name.upper()}_TYPE-")]
    args = []

    options = []

    for op_type in op_types:
        _, op_args, op_args_defaults = ga_operators[key][op_type]
        if op_args == []: continue

        arg_names = []
        arg_inputs = []

        for op_arg, op_arg_default in zip(op_args, op_args_defaults):
            _name, _input = single_value_option(f"{key}|{op_type}|{op_arg}", str(op_arg).capitalize(), None, op_arg_default)
            arg_names.append(_name)
            arg_inputs.append(_input)

        options.append([sg.Column([[sg.Column(arg_names), sg.Column(arg_inputs)]], expand_x=True, element_justification='c')])
    return combo_type, args, options

def tab():
    evo_type = [sg.Text("Select evolution type: ", pad=(10,None)), sg.Combo(evo_types, evo_types[0], font=FONT, size=(25,None), pad=(0,20,0,20), readonly=True, enable_events=True, key="-EVOTYPE_SELECT-")]

    sizes = [sg.Text("Generation count: ", pad=(10,None)), sg.Input("150", size=(5,None), enable_events=True, key="-GEN_COUNT-"),
             sg.Text("Starting population: "),             sg.Input("100", size=(5,None), enable_events=True, key="-POP_SIZE-")]

    selection_type, selection_args, selection_inputs = create_op_frame("Selection", "selection")
    selection = [sg.Frame("Selection", [selection_type, *selection_args, *selection_inputs], pad=(10,10), expand_x=True)]

    crossover_type, crossover_args, crossover_inputs = create_op_frame("Crossover", "crossover")
    crossover = [sg.Frame("Crossover ", [crossover_type, *crossover_args, *crossover_inputs], pad=(10,10), expand_x=True)]

    mutation_type, mutation_args, mutation_inputs = create_op_frame("Mutation", "mutation")
    mutation_probs = [[sg.Text("Mutation probabilities: ", pad=(10,None))],
                      [sg.Text("Individual: ", pad=(10, None)), sg.Input("0.75", font=FONT, size=(6,None), enable_events=True, key="-INDIV_MUT_PROB-"),
                       sg.Text("Action mutation: "),            sg.Input("0.1",  font=FONT, size=(6,None), enable_events=True, key="-ACT_MUT_PROB-"),
                       sg.Text("Body mutation: "),              sg.Input("0.1",  font=FONT, size=(6,None), enable_events=True, key="-BODY_MUT_PROB-")]]

    mutation = [sg.Frame("Mutation", [mutation_type, *mutation_args, *mutation_probs, *mutation_inputs], pad=(10,10), expand_x=True)]

    layout = [evo_type,
              sizes,
              selection,
              crossover,
              mutation]

    tab = sg.Tab("Evolution config", layout)
    return tab;

def handle_argument_inputs(window, values, key):
    out = ""
    for s in values[key]:
        if s in '1234567890.':
            out += s
    window[key].update(out)

    if "-POP_SIZE-" == key:
        window[f"NEATAgent|POP_SIZE"].update(values["-POP_SIZE-"])

def set_operator(operator_selected, window):
    op_type = ""
    if "selection" in operator_selected.lower():
        op_type = "selection"
    if "crossover" in operator_selected.lower():
        op_type = "crossover"
    if "mutation" in operator_selected.lower():
        op_type = "mutation"

    for key in list(window.AllKeysDict.keys()):
        if not isinstance(key, str): continue
        
        if key.startswith(op_type):
            window[key].update(visible= operator_selected in key)

    # # Hide other options 
    # agent_names = list(agents.keys())
    # for name in agent_names:
    #     window["options_"+name].update(visible=False)

    # # Show selected option
    # window["options_"+agent_selected].update(visible=True)

def events(window, event, values):
    if event == "-EVOTYPE_SELECT-":
        window['-EVOTYPE_SELECT-'].widget.select_clear()
        window.refresh()

    if event.startswith("-OP_"):
        set_operator(values[event], window)
        window[event].widget.select_clear()
        window.refresh()

    if "|" in event or \
       "-POP_SIZE-" == event or \
       "-GEN_COUNT-" == event or \
       "-INDIV_MUT_PROB-" == event or \
       "-ACT_MUT_PROB-" == event or \
       "-BODY_MUT_PROB-" == event:
        handle_argument_inputs(window,values,event)
        pass
