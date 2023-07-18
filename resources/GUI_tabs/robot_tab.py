#!/usr/bin/env python
"""GUI Robot tab

Robot tab is the second GUI tab used for selecting the robot with which we want
to do the experiment (see also :mod:`robots`). It comes with a selection menu,
image previewing the selected robot and small overview of the robot.
Optionally, if the robot has any body parts which can be altered during
evolution, the ``Select body parts for GA`` button will be enabled. When
clicked it is going to show a menu for configuring allowed body part ranges.
"""

import roboEvo
import resources.GUI_tabs.agent_tab as agent_tab
import resources.GUI_tabs.evo_tab as evo_tab 

import PySimpleGUI as sg
from PIL import Image, ImageTk
import numpy as np

DEFAULT_FONT = ("Arial", 14)

robots : 'dict[str, roboEvo.robots.BaseRobot]'
"""
Dictionary of all available robots by name. 

.. warning::
    User must include new custom robots inside this dictionary to make them
    available through GUI. The source code gives an example how to add new
    entries.
"""

robots = {robot.__class__.__name__ : robot for robot in [
    roboEvo.robots.AntV3(),
    roboEvo.robots.StickAnt(),
    roboEvo.robots.SpotLike(),
    roboEvo.robots.Walker2D(),
    roboEvo.robots.InvertedPendulum(),
    roboEvo.robots.InvertedDoublePendulum(),
]}

robot_names = list(robots.keys())

# for checking combo box changes
__current_robot = "AntV3"

def tab():
    options_menu = [sg.Text("Select robot: ", pad=(10,None)), sg.Combo(robot_names, robot_names[0], pad=(0,10), readonly=True, enable_events=True, key="-ROBOT_SELECT-")]

    img = sg.Image(source="", size=(400, 400), key="-ROBOT_IMAGE-")

    overview = sg.Frame("Robot overview", [[sg.Text(robots[robot_names[0]].description, size=(24, None), pad=(10,10), key="-ROBOT_OVERVIEW-")], 
                                           [sg.VPush()], 
                                           [sg.Push(), sg.Button("...", button_color=sg.theme_background_color(), border_width=0, key="-ROBOT_OVERVIEW_MORE-")]], expand_x=True, expand_y=True)

    body_changes = sg.Button("Select body parts for GA", key="-ROBOT_PARTS-")

    img_overview = [sg.Column([[img]]), sg.Column([[overview],[body_changes]], expand_x=True, expand_y=True)]

    main = [[sg.VPush()],
            options_menu,
            img_overview,
            [sg.VPush()]]

    tab = sg.Tab("Robot select", main, key="-ROBOT_TAB-")
    return tab

def set_robot(robot_selected, window, agent=None):
    robot = robots[robot_selected]

    im = Image.open(robot.picture_path)
    crop = im.crop((250,50,650,445))
    image = ImageTk.PhotoImage(image=crop)
    window['-ROBOT_IMAGE-'].update(data=image)

    # check needed range
    cutoff = 320
    
    TEXT = robot.description
    if len(TEXT) > cutoff:
        TEXT = TEXT[:cutoff] + " ..."

    window["-ROBOT_OVERVIEW_MORE-"].update(visible=len(TEXT)>cutoff)
    window["-ROBOT_OVERVIEW-"].update(TEXT)
    window["-ROBOT_PARTS-"].update(disabled = not (len(robot.body_parts) > 0)) # if robot has specified vars in XML, enable button

    agent_tab.reload_agents(window, robot, agent)

def popup_robot_parts(robot_selected, agents, agent_selected, window, values):

    def handle_argument_inputs(window, values, key):
        out = ""
        dot = False
        for s in values[key]:
            if s == '-' and out == "":
                out += '-'
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

    robot = robots[robot_selected]
    agent = agents[agent_selected]
    body_parts = np.array(list(robot.body_parts.keys()))

    # unlocked_body_parts = np.array(agent.orig_body_part_mask)
    unlocked_mask = [True if mask else False for mask in agent.orig_body_part_mask]

    title = sg.Text("Unlock body parts to enable their changes during GA run and select their allowed range of values", size=(60,2))

    names = []
    buttons = []
    inputs = []
    for i, body_part in enumerate(body_parts):
        unlocked = unlocked_mask[i]

        default_min = agent.orig_body_part_mask[i][0] if unlocked else robot.body_parts[body_part]
        default_max = agent.orig_body_part_mask[i][1] if unlocked else robot.body_parts[body_part]

        input = [sg.Text("MIN", font=(DEFAULT_FONT[0], 10),pad=(10,8)), sg.Input(default_min, size=(6,None),pad=(10,8), enable_events=True, key=body_part+"_min", disabled=not unlocked),
                 sg.Text("MAX", font=(DEFAULT_FONT[0], 10),pad=(10,8)), sg.Input(default_max, size=(6,None),pad=(10,8), enable_events=True, key=body_part+"_max", disabled=not unlocked)]

        names.append([sg.Text(body_part,pad=(0,8))])
        buttons.append([sg.Button("UNLOCKED" if unlocked else "LOCKED", size=(9,1), key=f"{body_part}|switch")])
        inputs.append(input)

    column_layout = [[sg.Column(names), sg.Column(buttons), sg.Column(inputs)]]

    layout = [[title],
              [sg.Column(column_layout, expand_x=True, element_justification='c')]]

    popup = sg.Window("Select body parts for GA", layout, size=(700,400), font=DEFAULT_FONT, keep_on_top=True, modal=True)
    out_popup_values = None
    while True:
        event, popup_values = popup.read()
        if event == sg.WIN_CLOSED or event == 'Exit':
            break

        if "switch" in event:
            bp = event.split('|').pop(0)
            index = int(np.argwhere(body_parts == bp))
            unlocked_mask[index] = not unlocked_mask[index]

            popup[event].update("UNLOCKED" if unlocked_mask[index] else "LOCKED")
            popup[f"{bp}_min"].update(disabled = not unlocked_mask[index])
            popup[f"{bp}_max"].update(disabled = not unlocked_mask[index])

        if "_min" in event or "_max" in event:
            handle_argument_inputs(popup, popup_values, event)
            handle_range_inputs(popup, popup_values, event)

        out_popup_values = popup_values

        popup.refresh()

    if out_popup_values == None:
        return

    body_part_mask = []
    for i, bp in enumerate(body_parts):
        if unlocked_mask[i]:
            min = out_popup_values[f"{bp}_min"]
            max = out_popup_values[f"{bp}_max"]
            min = 0.0 if min in ["", "-", ".", "-."] else float(min)
            max = 0.0 if max in ["", "-", ".", "-."] else float(max)

            body_part_mask.append((min, max))
        else:
            body_part_mask.append(False)

    agent = agent.__class__(robot, body_part_mask, evo_tab.evo_types[values["-EVO_TYPE_SELECT-"]], gui=True)
    agent_tab.reload_agents(window, robot, agent)

def expand_description(text):
    frame = sg.Frame("Description", [[sg.Text(text, size=(60,None), pad=(10,10))]])
    sg.Window("Description long", [[frame]], font=DEFAULT_FONT, keep_on_top=True, modal=True).read(close=True)

def events(window, event, values, agents):
    if event == "-ROBOT_SELECT-":
        global __current_robot
        if values[event] == __current_robot: return
        __current_robot = values[event]
        set_robot(values[event], window)
        window[event].widget.select_clear()
        window.refresh()

    if event == "-ROBOT_PARTS-":
        popup_robot_parts(values['-ROBOT_SELECT-'], agents, values['-AGENT_SELECT-'], window, values)
            
    if event == "-ROBOT_OVERVIEW_MORE-": 
        expand_description(robots[values['-ROBOT_SELECT-']].description)

