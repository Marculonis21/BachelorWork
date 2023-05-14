#!/usr/bin/env python

import PySimpleGUI as sg

from experiment_setter import Experiments
experiments = Experiments()

font = ("Helvetica", 15)

def tab():
    tab = sg.Tab("Evolution config", [[]])
    return tab;

def events(window, event, values):
    pass
