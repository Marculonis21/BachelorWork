#!/usr/bin/env python

import PySimpleGUI as sg

import roboEvo

from typing import Dict

font = ("Helvetica", 15)

evo_types = [e_type.name for e_type in roboEvo.gaAgents.EvoType]

def tab():
    evo_type = [sg.Text("Select evolution type: ", pad=(10,None)), sg.Combo(evo_types, evo_types[0], size=(25,None), pad=(0,20,0,20), readonly=True, enable_events=True, key="-EVOTYPE_SELECT-")]

    sizes = [sg.Text("Generation count: ", pad=(10,None)), sg.Input("150", size=(5,None), enable_events=True, key="-GEN_COUNT-"),
             sg.Text("Starting population: "),             sg.Input("100", size=(5,None), enable_events=True, key="-POP_SIZE-")]

    mutation_probs = [[sg.Text("Mutation probabilities: ", pad=(10,None))],
                      [sg.Text("Individual: ", pad=(10, None)), sg.Input("0.75", size=(6,None), enable_events=True, key="-INDIV_MUT_PROB-"),
                       sg.Text("Action mutation: "),            sg.Input("0.1",  size=(6,None), enable_events=True, key="-ACT_MUT_PROB-"),
                       sg.Text("Body mutation: "),              sg.Input("0.1",  size=(6,None), enable_events=True, key="-BODY_MUT_PROB-")]]

    layout = [evo_type,
              sizes,
              *mutation_probs]

    tab = sg.Tab("Evolution config", layout)
    return tab;

def events(window, event, values):
    if event == "-EVOTYPE_SELECT-":
        window['-EVOTYPE_SELECT-'].widget.select_clear()
        window.refresh()
