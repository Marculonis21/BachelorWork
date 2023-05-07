#!/usr/bin/env python

import roboEvo

import PySimpleGUI as sg
from PIL import Image, ImageTk
import numpy as np

font = ("Helvetica", 15)

robots : 'dict[str, roboEvo.robots.BaseRobot]'
robots = {"OpenAI Ant-v3" : roboEvo.robots.AntV3(),
          "Basic Ant"     : roboEvo.robots.StickAnt(),
          "SpotLike dog"  : roboEvo.robots.SpotLike()}

robot_names = list(robots.keys())

def tab():
    # options_menu = [sg.Text("Select robot: "), sg.OptionMenu(robot_names, robot_names[0], pad=(0,10), key="-ROBOT_SELECT-")]
    options_menu = [sg.Text("Select robot: "), sg.Combo(robot_names, robot_names[0], pad=(0,10), readonly=True, enable_events=True, key="-ROBOT_SELECT-")]

    img = sg.Image(source="", size=(400, 400), key="-ROBOT_IMAGE-")

    overview = sg.Frame("Robot overview", [[sg.Text(robots[robot_names[0]].description,font=("Helvetica", 14), size=(24, None), pad=(10,10), key="-ROBOT_OVERVIEW-")], 
                                           [sg.VPush()], 
                                           [sg.Push(), sg.Button("...", button_color=sg.theme_background_color(), border_width=0, key="-ROBOT_OVERVIEW_MORE-")]], expand_x=True, expand_y=True)

    body_changes = sg.Button("Select body parts for GA", key="-ROBOT_PARTS-")

    # img_overview = [sg.Column([[img, overview,]], background_color='red', expand_x=True, expand_y=True)]
    img_overview = [sg.Column([[img]]), sg.Column([[overview],[body_changes]], expand_x=True, expand_y=True)]

    main = [[sg.VPush()],
            options_menu,
            img_overview,
            [sg.VPush()]]

    tab = sg.Tab("Robot select", main)
    return tab

def set_robot(robot_selected, window, agents, agent=None):
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

    agents["Full Random"]              = roboEvo.gaAgents.FullRandomAgent(   robot, [False]*len(robot.body_parts) if agent == None else agent.body_part_mask, 25)
    agents["Sine Function Full"]       = roboEvo.gaAgents.SineFuncFullAgent( robot, [False]*len(robot.body_parts) if agent == None else agent.body_part_mask)
    agents["Sine Function Half"]       = roboEvo.gaAgents.SineFuncHalfAgent( robot, [False]*len(robot.body_parts) if agent == None else agent.body_part_mask)
    agents["Step Cycle Half"]          = roboEvo.gaAgents.StepCycleHalfAgent(robot, [False]*len(robot.body_parts) if agent == None else agent.body_part_mask, 20)
    agents["Truncated Fourier Series"] = roboEvo.gaAgents.TFSAgent(          robot, [False]*len(robot.body_parts) if agent == None else agent.body_part_mask)

    if agent != None:
        pass

def popup_robot_parts(robot_selected, agents, agent_selected):
    robot = robots[robot_selected]
    agent = agents[agent_selected]
    body_parts = np.array(list(robot.body_parts.keys()))

    unlocked_body_parts = np.array(agent.body_part_mask)

    title = sg.Text("Unlock body parts to enable their changes during GA run", size=(None,2))

    left   = [[sg.Text("Locked body parts")],
              [sg.Listbox(body_parts[~unlocked_body_parts], select_mode=sg.LISTBOX_SELECT_MODE_EXTENDED, font=("Helvetica", 12), expand_x=True, expand_y=True, no_scrollbar=True, pad=(10,10), key="LOCKED")]]
    middle = [[sg.VPush()],
              [sg.Button(">>>", key="ADD")],
              [sg.Button("<<<", key="REMOVE")],
              [sg.VPush()]]
    right  = [[sg.Text("Unlocked body parts")],
              [sg.Listbox(body_parts[unlocked_body_parts], select_mode=sg.LISTBOX_SELECT_MODE_EXTENDED, font=("Helvetica", 12), expand_x=True, expand_y=True, no_scrollbar=True, pad=(10,10), key="UNLOCKED")]]

    layout = [[title],
              [sg.Column(left, expand_x=True, expand_y=True, element_justification='c'), 
               sg.Column(middle, expand_y=True, element_justification='c'),
               sg.Column(right, expand_x=True, expand_y=True, element_justification='c')]]

    popup = sg.Window("Select body parts for GA", layout, size=(700,400), font=font, keep_on_top=True, modal=True)
    while True:
        event, values = popup.read()

        if event == sg.WIN_CLOSED or event == 'Exit':
            break

        if event == "ADD":
            selected = values["LOCKED"]
            if len(selected) == 0:
                continue

            indexes = np.isin(body_parts, selected)

            unlocked_body_parts[indexes] = True

        if event == "REMOVE":
            selected = values["UNLOCKED"]
            if len(selected) == 0:
                continue

            indexes = np.isin(body_parts, selected)

            unlocked_body_parts[indexes] = False

        popup["LOCKED"].update(body_parts[~unlocked_body_parts])
        popup["UNLOCKED"].update(body_parts[unlocked_body_parts])
        popup.refresh()

def expand_description(text):
    frame = sg.Frame("Description", [[sg.Text(text, size=(60,None), font=("Helvetica", 14), pad=(10,10))]])
    sg.Window("Description long", [[frame]], font=font, keep_on_top=True, modal=True).read(close=True)

def events(window, event, values, agents):
    if event == "-ROBOT_SELECT-":
        set_robot(values['-ROBOT_SELECT-'], window, agents)
        window['-ROBOT_SELECT-'].widget.select_clear()
        window.refresh()

    if event == "-ROBOT_PARTS-":
        popup_robot_parts(values['-ROBOT_SELECT-'], agents, values['-AGENT_SELECT-'])
            
    if event == "-ROBOT_OVERVIEW_MORE-": 
        expand_description(robots[values['-ROBOT_SELECT-']].description)

