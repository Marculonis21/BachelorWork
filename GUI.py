#!/usr/bin/env python

import PySimpleGUI as sg
from PIL import Image, ImageTk
from distributed.batched import gen
import antGA
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading

use_custom_titlebar = False
font = ("Helvetica", 15)

def main_tab():
    frame_text = [[sg.Text("", font=("Helvetica", 16), size=(72, 8), pad=(20,20), key="-MAIN_FRAME_TEXT-")]]

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

robot_names = ["OpenAi Ant", "Basic Ant"]
robot_overview = {robot_names[0]:"",
                  robot_names[1]:""}
robot_pics = {robot_names[0]:"Ant-v3",
              robot_names[1]:"Basic-Ant"}

def set_robot_image(index_selected):
    im = Image.open("./docs/UI/UIdata/"+robot_pics[robot_names[index_selected]])
    crop = im.crop((250,50,650,445))
    image = ImageTk.PhotoImage(image=crop)
    window['-ROBOT_IMAGE-'].update(data=image)
                
def robot_select_tab():
    options_menu = [sg.Text("Select robot: "), sg.OptionMenu(robot_names, robot_names[0], pad=(0,10), key="-ROBOT_SELECT-")]

    img = sg.Image(source="", size=(400, 400), key="-ROBOT_IMAGE-")
    TEXT = "TEXT TEXT TEXT"
    overview = sg.Frame("Robot overview", [[sg.Text(TEXT,font=("Helvetica", 16),pad=(20,20), key="-ROBOT_OVERVIEW-")], 
                                           [sg.VPush()], 
                                           [sg.Push(), sg.Button("...", button_color=sg.theme_background_color(), border_width=0, key="-ROBOT_OVERVIEW_MORE-")]], expand_x=True, expand_y=True)

    img_overview = [sg.Column([[img, overview]], expand_x=True, expand_y=True)]

    main = [[sg.VPush()],
            options_menu,
            img_overview,
            [sg.VPush()]]

    tab = sg.Tab("Robot Select", main)
    return tab;

agent_types = ["Full Random", "Sine Function Full", "Sine Function Half", "Step Cycle Half"]
agent_overview = {"Full Random"        : "Full Random agent\n    Starts off as a sequence of random actions for each motor for chosen amount of steps. Behavior of the agent is then made by repeating this sequence till end state is reached periodically.",
                  "Sine Function Full" : "Sine Function Full agent\n    Each motor of the robot is controlled by sine wave. Values of these agents are made of only 4 parameters (amplitude, frequency, shiftX, shiftY) for each motor.",
                  "Sine Function Half" : "Sine Function Full agent\n    Similar to Sine Function Full agent, however only half of robot's motors are controlled by sine waves. The other half is symmetrical (point symmetry through center of the body).",
                  "Step Cycle Half"    : "Step Cycle Half agent\n    Combination of random and half agent. STEPCOUNT long sequences of random actions for half of the motors are created and and then by symmetry transfered to opposing motors. During runtime, sequences of actions are repeatedly performed"}

def set_agent_overview(agent_selected):
    TEXT = agent_overview[agent_selected]
    if len(TEXT) > 260:
        TEXT = TEXT[:260] + " ..."

    window["-AGENT_OVERVIEW_MORE-"].update(visible=len(TEXT)>260)
    window["-AGENT_OVERVIEW-"].update(TEXT)

def agent_select_tab():
    options_menu = [sg.Text("Select agent type: "), sg.OptionMenu(agent_types, agent_types[0], pad=(0,20,0,20), key="-AGENT_SELECT-")]
    frame = [sg.Frame("Agent overview", [[sg.Text(agent_overview[agent_types[0]], font=("Helvetica", 16), size=(72, 6), pad=(20,20), key="-AGENT_OVERVIEW-")],
                                         [sg.Push(), sg.Button("...", button_color=sg.theme_background_color(), border_width=0, key="-AGENT_OVERVIEW_MORE-")]], expand_x=True, pad=(10,0))]

    tab = sg.Tab("Agent config", [options_menu, frame])
    return tab;

def evolution_select_tab():
    tab = sg.Tab("Evolution config", [[]])
    return tab;

def make_window():
    tabGroup = [[sg.TabGroup([[main_tab(), robot_select_tab(), agent_select_tab()]], size=(800,600))]]

    window = sg.Window('Test GUI', tabGroup, size=(800,600), font=font, finalize=True, keep_on_top=True)
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

    TEXT = '''- Robot selected: {}\n- Robot controlling agent: {}'''.format(values['-ROBOT_SELECT-'], values["-AGENT_SELECT-"])
    window["-MAIN_FRAME_TEXT-"].update(TEXT)

    set_robot_image(0)
    set_agent_overview(agent_types[0])

    window_values = {}

# Event Loop to process "events" and get the "values" of the inputs
    SETUP = True
    RUN = True
    while SETUP:
        event, values = window.read()
        # print(event,values)

        if event == sg.WIN_CLOSED or event == 'Exit':
            quit()
            break
        
        if event == "-ROBOT_SELECT-":
            set_robot_image(robot_names.index(values['-ROBOT_SELECT-']))
            window.refresh()

        if event == "-AGENT_SELECT-":
            set_agent_overview(values["-AGENT_SELECT-"])
            window.refresh()

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


        TEXT = '''- Robot selected: {}\n- Robot controlling agent: {}'''.format(values['-ROBOT_SELECT-'], values["-AGENT_SELECT-"])
        window["-MAIN_FRAME_TEXT-"].update(TEXT)

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
    def drawChart():
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

    def updateChart():
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
        robot = max_steps = agent = show_best = save_best = save_dir = None
        if window_values["-ROBOT_SELECT-"] == "OpenAi Ant":
            robot = "Ant-v3"
        if window_values["-ROBOT_SELECT-"] == "Basic Ant":
            robot = "CustomAntLike-v1"

        agent = window_values["-AGENT_SELECT-"]
        show_best = window_values["-CB_FINAL-"]
        save_best = window_values["-CB_SAVEBEST-"]
        save_dir = window_values["Browse"]

        max_steps = 500
        return  robot, max_steps, agent, show_best, save_best, save_dir
            

    def startRun():
        robot, max_steps, agent, show_best, save_best, save_dir = GetParams()
        antGA.RunFromGui(robot, max_steps, agent, show_best, save_best, save_dir)

    working_thread = threading.Thread(target=startRun, daemon=True)
    working_thread.start()

    drawChart();

    while RUN:
        if working_thread.is_alive():
            event, values = window.read(timeout=2500)
        else:
            window["-EXIT-"].update(button_color="green")
            event, values = window.read()
        # print(event)

        if event == sg.WIN_CLOSED or event == 'Exit':
            break

        if event == "-RUN_PREVIEW-":
            antGA.RaisePreview()

        if event == "-EXIT-":
            antGA.RaiseAbort()
            working_thread.join()
            window.close()

        updateChart()

        window["-GENNUM-"].update(str(antGA.GUI_GEN_NUMBER))
        if len(antGA.GUI_FITNESS) > 0:
            window["-MEANFIT-"].update(str(antGA.GUI_FITNESS[0]))
            window["-MINFIT-"].update(str(antGA.GUI_FITNESS[1]))
            window["-MAXFIT-"].update(str(antGA.GUI_FITNESS[2]))

    window.close()
