#!/usr/bin/env python

import PySimpleGUI as sg
from PIL import Image, ImageTk
import antGA
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import numpy as np

font = ("Helvetica", 15)

def main_tab():
    frame_text = [[sg.Text("", font=("Helvetica", 14), size=(58, 8), pad=(10,10), key="-MAIN_SETTINGS_OVERVIEW-")]]

    frame = [sg.Frame("Settings overview", frame_text, size=(800, 300), pad=(10,10))]

    options_l = [[sg.Input("250", size=(5,None), enable_events=True, key="-MAIN_GEN_COUNT_IN-"), sg.Text("Generation count")],
                 [sg.Checkbox("Show final run", key="-CB_FINAL-")],
                 [sg.Checkbox("Show progress runs", key="-CB_PROGRESS-")],
                 [sg.Sizer(0, 1)]]

    options_r = [[sg.Input("50", size=(5,None), enable_events=True, key="-MAIN_POP_SIZE_IN-"), sg.Text("Starting population")],
                 [sg.Checkbox("Save best agent", default=True, key="-CB_SAVEBEST-")],
                 [sg.Text("Save directory:"), sg.Text(".", size=(30,None), font=("Helvetica", 10), key="-SAVE_DIR-")],
                 [sg.FolderBrowse("Browse", target="-SAVE_DIR-")]]

    options = [sg.Col([[sg.Col(options_l, element_justification='l', vertical_alignment='t', expand_x=True), sg.Col(options_r, element_justification='l', vertical_alignment='t', expand_x=True)]], expand_x=True, pad=(5,10))]

    start = [sg.Push(), sg.Button("Start", key="-START-")]

    main = [frame,
            options,
            [sg.VPush()],
            start]

    tab = sg.Tab("Main", main)
    return tab

def robot_select_callback(var, index, mode):
    window.write_event_value('-ROBOT_SELECT-', window['-ROBOT_SELECT-'].TKStringVar.get())

def agent_select_callback(var, index, mode):
    window.write_event_value('-AGENT_SELECT-', window['-AGENT_SELECT-'].TKStringVar.get())

def expand_description(text):
    frame = sg.Frame("Description", [[sg.Text(text, size=(60,None), font=("Helvetica", 14), pad=(10,10))]])
    sg.Window("Description long", [[frame]], font=font, keep_on_top=True, modal=True).read(close=True)

import robots.robots as robotsClass
robots = {"OpenAI Ant-v3" : robotsClass.AntV3(),
          "Basic Ant"     : robotsClass.StickAnt(),
          "SpotLike dog" : robotsClass.SpotLike()}

body_part_mask = []

def popup_robot_parts(robot_selected):
    global body_part_mask

    robot = robots[robot_selected]
    body_parts = np.array(list(robot.body_parts.keys()))

    body_part_mask = np.array(body_part_mask)
    neg_mask = np.logical_not(body_part_mask)

    title = sg.Text("Unlock body parts to enable their changes during GA run", size=(None,2))

    left   = [[sg.Text("Locked body parts")],
              [sg.Listbox(body_parts[neg_mask], select_mode=sg.LISTBOX_SELECT_MODE_EXTENDED, font=("Helvetica", 12), expand_x=True, expand_y=True, no_scrollbar=True, pad=(10,10), key="LOCKED")]]
    middle = [[sg.VPush()],
              [sg.Button(">>>", key="ADD")],
              [sg.Button("<<<", key="REMOVE")],
              [sg.VPush()]]
    right  = [[sg.Text("Unlocked body parts")],
              [sg.Listbox(body_parts[body_part_mask], select_mode=sg.LISTBOX_SELECT_MODE_EXTENDED, font=("Helvetica", 12), expand_x=True, expand_y=True, no_scrollbar=True, pad=(10,10), key="UNLOCKED")]]

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

            body_part_mask[indexes] = True
            neg_mask = np.logical_not(body_part_mask)

        if event == "REMOVE":
            selected = values["UNLOCKED"]
            if len(selected) == 0:
                continue

            indexes = np.isin(body_parts, selected)

            body_part_mask[indexes] = False
            neg_mask = np.logical_not(body_part_mask)

        popup["LOCKED"].update(body_parts[neg_mask])
        popup["UNLOCKED"].update(body_parts[body_part_mask])
        popup.refresh()

def set_robot(robot_selected):
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

    global body_part_mask
    # reset body part mask
    body_part_mask = [False for _ in range(len(robots[robot_selected].body_parts))]
                
def robot_select_tab():
    robot_names = list(robots.keys())
    options_menu = [sg.Text("Select robot: "), sg.OptionMenu(robot_names, robot_names[0], pad=(0,10), key="-ROBOT_SELECT-")]

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

    tab = sg.Tab("Robot Select", main)
    return tab;

import gaAgent
agents : 'dict[str, gaAgent.AgentType]'
agents = {"Full Random" : gaAgent.FullRandomAgent.ForGUI(),
          "Sine Function Full" : gaAgent.SineFuncFullAgent.ForGUI(),
          "Sine Function Half" : gaAgent.SineFuncHalfAgent.ForGUI(),
          "Step Cycle Half" : gaAgent.StepCycleHalfAgent.ForGUI(),}


def single_value_option(key, text, default, disabled=False):
    return [sg.Text(text,pad=(10,10)), sg.Input(default, size=(8,None), enable_events=True, key=key, disabled=disabled)]

def range_value_option(key, text, default_min, default_max, disabled=False):
    return [sg.Text(text,pad=(10,10)), sg.Text("MIN", font=("Helvetica", 12)), sg.Input(default_min, size=(8,None), enable_events=True, key=key+"_min", disabled=disabled), 
                                       sg.Text("MAX", font=("Helvetica", 12)), sg.Input(default_max, size=(8,None), enable_events=True, key=key+"_max", disabled=disabled)]

import math
agent_argument_options = [sg.Column([single_value_option("cycle_repeat", "Step Count", 25)], 
                                    element_justification='c', expand_x=True, visible=False, key="options_Full Random"),
                          sg.Column([range_value_option("amplitude_range", "Amplitude range", 0.5, 5),
                                     range_value_option("frequency_range", "Frequency range", 0.5, 5),
                                     range_value_option("shift_x_range", "Shift x", 0, 2*math.pi),
                                     range_value_option("shift_y_range", "Amplitude range", "gaAgent.py", "gaAgent.py", disabled=True)], 
                                    element_justification='c', expand_x=True, visible=False, key="options_Sine Function Full"),
                          sg.Column([range_value_option("amplitude_range", "Amplitude range", 0.5, 5),
                                     range_value_option("frequency_range", "Frequency range", 0.5, 5),
                                     range_value_option("shift_x_range", "Shift x", 0, 2*math.pi),
                                     range_value_option("shift_y_range", "Amplitude range", "gaAgent.py", "gaAgent.py", disabled=True)],
                                    element_justification='c', expand_x=True, visible=False, key="options_Sine Function Half"),
                          sg.Column([single_value_option("cycle_repeat", "Step Count", 25)], 
                                    element_justification='c', expand_x=True, visible=False, key="options_Step Cycle Half"),
                          ]

def set_agent(agent_selected):
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

    window["options_"+agent_selected].update(visible=True)

def agent_select_tab():
    agent_names = list(agents.keys())
    options_menu = [sg.Text("Select agent type: "), sg.OptionMenu(agent_names, agent_names[0], pad=(0,20,0,20), key="-AGENT_SELECT-")]
    frame = [sg.Frame("Agent overview", [[sg.Text(agents[agent_names[0]].description, font=("Helvetica", 14), size=(58, 6), pad=(10,10), key="-AGENT_OVERVIEW-")],
                                         [sg.Push(), sg.Button("...", button_color=sg.theme_background_color(), border_width=0, key="-AGENT_OVERVIEW_MORE-")]], expand_x=True, pad=(10,0))]

    tab = sg.Tab("Agent config", [options_menu, 
                                  frame,
                                  [sg.VPush()],
                                  agent_argument_options,
                                  [sg.VPush()]])
    return tab;

def evolution_config_tab():
    tab = sg.Tab("Evolution config", [[]])
    return tab;

def make_window():
    tabGroup = [[sg.TabGroup([[main_tab(), robot_select_tab(), agent_select_tab(), evolution_config_tab()]], size=(800,600))]]

    window = sg.Window('Test GUI', tabGroup, size=(800,600), font=font, finalize=True,  use_default_focus=False)
    window['-ROBOT_SELECT-'].TKStringVar.trace("w", robot_select_callback)
    window['-AGENT_SELECT-'].TKStringVar.trace("w", agent_select_callback)
    window['-AGENT_OVERVIEW_MORE-'].block_focus()
    window['-ROBOT_OVERVIEW_MORE-'].block_focus()

    return window

def make_run_window():
    layout = [[sg.Canvas(key='-FIG-')],
              [sg.Text("Gen Number: "), sg.Text("", key="-GENNUM-")],
              [sg.Text("Max Fit: "), sg.Text("", key="-MAXFIT-")], 
              [sg.Text("Min Fit: "), sg.Text("", key="-MINFIT-")], 
              [sg.Text("Mean Fit: "), sg.Text("", key="-MEANFIT-"), sg.Push(), sg.Button("Preview best", key="-RUN_PREVIEW-"), sg.Button("Exit", button_color='red', mouseover_colors='orange red', key="-EXIT-")]]

    window = sg.Window('Process running', layout, font=font, finalize=True, keep_on_top=True)

    return window

if __name__ == "__main__":
# Create the Window
    window = make_window()
    _, values = window.read(timeout=0)

    agent_names = list(agents.keys())
    robot_names = list(robots.keys())
    set_robot(robot_names[0])
    set_agent(agent_names[0])

    TEXT = '''- Robot selected: {}\n- Robot controlling agent: {}'''.format(values['-ROBOT_SELECT-'], values["-AGENT_SELECT-"])
    window["-MAIN_SETTINGS_OVERVIEW-"].update(TEXT)

    window_values = {}

# Event Loop to process "events" and get the "values" of the inputs
    SETUP = True
    RUN = True
    while SETUP:
        event, values = window.read()
        # print(event,values)

        if event == sg.WIN_CLOSED or event == 'Exit':
            quit()
        
        if event == "-ROBOT_SELECT-":
            set_robot(values['-ROBOT_SELECT-'])
            window.refresh()

        if event == "-ROBOT_PARTS-":
            popup_robot_parts(values['-ROBOT_SELECT-'])
            # window.refresh()

        if event == "-AGENT_SELECT-":
            set_agent(values["-AGENT_SELECT-"])
            window.refresh()

        if event == "-ROBOT_OVERVIEW_MORE-" or event == "-AGENT_OVERVIEW_MORE-":
            if event == "-ROBOT_OVERVIEW_MORE-": expand_description(robots[values['-ROBOT_SELECT-']].description)
            if event == "-AGENT_OVERVIEW_MORE-": expand_description(agents[values['-AGENT_SELECT-']].description)

        if (event == '-MAIN_GEN_COUNT_IN-' and values['-MAIN_GEN_COUNT_IN-']) or \
           (event == '-MAIN_POP_SIZE_IN-' and values['-MAIN_POP_SIZE_IN-']): 
            gen_count = ""
            pop_size = ""
            for s in values['-MAIN_GEN_COUNT_IN-']:
                if s in '1234567890':
                    gen_count += s

            for s in values['-MAIN_POP_SIZE_IN-']:
                if s in '1234567890':
                    pop_size += s

            window['-MAIN_GEN_COUNT_IN-'].update(gen_count)
            window['-MAIN_POP_SIZE_IN-'].update(pop_size)


        robot_selected = values['-ROBOT_SELECT-']
        agent_selected = values['-AGENT_SELECT-']

        TEXT = "" #'''- Robot selected: {}\n- Robot controlling agent: {}'''.format(values['-ROBOT_SELECT-'], values["-AGENT_SELECT-"])
        TEXT += "- Robot selected: {}\n".format(robot_selected)

        for i, part in enumerate(robots[robot_selected].body_parts):
            if i == 0:
                TEXT += "    - Body parts for GA:\n"
            TEXT += "        - {} ... {}\n".format(part, "unlocked" if body_part_mask[i] else "locked")
        TEXT += "\n"

        TEXT += "- Robot controlling agent: {}\n".format(agent_selected)

        window["-MAIN_SETTINGS_OVERVIEW-"].update(TEXT)

        if event == "-START-":
            window_values = values
            window.close()
            break;

    window = make_run_window()
    SETUP = False

    def draw_figure(canvas, figure):
        figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
        figure_canvas_agg.draw_idle()
        figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
        return figure_canvas_agg

    FIG = None
    FIG_AGG = None
    def draw_chart():
        global FIG, FIG_AGG
        FIG = plt.figure()
        plt.title('Training')
        plt.xlabel('Episode')
        plt.ylabel('Fitness')
        plt.plot(antGA.GUI_GRAPH_VALUES[0], label='Mean')
        plt.plot(antGA.GUI_GRAPH_VALUES[1], label='Min')
        plt.plot(antGA.GUI_GRAPH_VALUES[2], label='Max')
        plt.legend(loc='upper left', fontsize=9)
        plt.tight_layout()
        FIG_AGG = draw_figure(
            window['-FIG-'].TKCanvas, FIG)

    def update_chart():
        global FIG, FIG_AGG
        FIG_AGG.get_tk_widget().forget()
        plt.clf()
        # plt.cla()
        plt.title('Training')
        plt.xlabel('Episode')
        plt.ylabel('Fitness')
        plt.plot(antGA.GUI_GRAPH_VALUES[0], label='Mean')
        plt.plot(antGA.GUI_GRAPH_VALUES[1], label='Min')
        plt.plot(antGA.GUI_GRAPH_VALUES[2], label='Max')
        plt.legend(loc='upper left', fontsize=9)
        plt.tight_layout()

        FIG_AGG = draw_figure(
            window['-FIG-'].TKCanvas, FIG)

    def GetParams():
        robot = agent = population_size = generation_count = show_best = save_best = save_dir = None

        robot = robots[window_values["-ROBOT_SELECT-"]] 

        agent_selected = window_values["-AGENT_SELECT-"]
        if agent_selected == "Full Random":          agent = gaAgent.FullRandomAgent(robot, body_part_mask, 25)
        elif agent_selected == "Sine Function Full": agent = gaAgent.SineFuncFullAgent(robot, body_part_mask)
        elif agent_selected == "Sine Function Half": agent = gaAgent.SineFuncHalfAgent(robot, body_part_mask)
        elif agent_selected == "Step Cycle Half":    agent = gaAgent.StepCycleHalfAgent(robot, body_part_mask, 20)
        else: raise AttributeError("Unknown control agent type - " + agent_selected)

        population_size = window_values["-MAIN_POP_SIZE_IN-"]
        generation_count = window_values["-MAIN_GEN_COUNT_IN-"]

        show_best = window_values["-CB_FINAL-"]
        save_best = window_values["-CB_SAVEBEST-"]
        save_dir = window_values["Browse"]

        return  robot, agent, population_size, generation_count, show_best, save_best, save_dir
            
    def startRun():
        robot, agent, population_size, generation_count, show_best, save_best, save_dir = GetParams()
        antGA.RunFromGui(robot, agent, population_size, generation_count, show_best, save_best, save_dir)

    working_thread = threading.Thread(target=startRun, daemon=True)
    working_thread.start()

    draw_chart();

    while RUN:
        if working_thread.is_alive():
            event, values = window.read(timeout=2500)
        else:
            window["-EXIT-"].update(button_color="green")
            event, values = window.read()
            break
        # print(event)

        if event == sg.WIN_CLOSED or event == 'Exit':
            break

        if event == "-RUN_PREVIEW-":
            antGA.RaisePreview()

        if event == "-EXIT-":
            antGA.RaiseAbort()
            working_thread.join()
            window.close()

        update_chart()

        window["-GENNUM-"].update(str(antGA.GUI_GEN_NUMBER))
        if len(antGA.GUI_FITNESS) > 0:
            window["-MEANFIT-"].update(str(antGA.GUI_FITNESS[0]))
            window["-MINFIT-"].update(str(antGA.GUI_FITNESS[1]))
            window["-MAXFIT-"].update(str(antGA.GUI_FITNESS[2]))

    window.close()
