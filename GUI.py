#!/usr/bin/env python

import PySimpleGUI as sg
# from PIL import Image
import antGA
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading

use_custom_titlebar = False
font = ("Helvetica", 15)

def main_tab():
    frame_layout = [[sg.Text("", font=("Helvetica", 16), size=(100, 20), pad=(20,20), key="-FRAME_TEXT-")]]

    frame = [[sg.Frame("Settings overview", frame_layout, size=(800, 400))]]

    options_l = [[sg.Checkbox("Show final run", key="-CB_FINAL-")],
                 [sg.Checkbox("Show progress runs", key="-CB_PROGRESS-")],
                 [sg.Sizer(0, 1)]]

    options_r = [[sg.Checkbox("Save best agent", default=True, key="-CB_SAVEBEST-")],
                 [sg.Text("Save directory:"), sg.Text(".", size=(30,None), key="-SAVE_DIR-")],
                 [sg.FolderBrowse("Browse", target="-SAVE_DIR-")]]

    options = [[sg.Col([[sg.Col(options_l, element_justification='l', vertical_alignment='t', expand_x=True), sg.Col(options_r, element_justification='l', vertical_alignment='t', expand_x=True)]], expand_x=True)]]

    start = [[sg.Col([[sg.Button("Start", key="-START-")]], expand_x=True, element_justification='r')]]

    main = frame + options + start

    tab = sg.Tab("Main", main)
    return tab

def robot_select_callback(var, index, mode):
    window.write_event_value('-ROBOT_SELECT-', window['-ROBOT_SELECT-'].TKStringVar.get())
def agent_select_callback(var, index, mode):
    window.write_event_value('-AGENT_SELECT-', window['-AGENT_SELECT-'].TKStringVar.get())


robot_names = ["OpenAi Ant", "Basic Ant", "TEST"]
robot_pics = {robot_names[0]:"Ant-v3",
              robot_names[1]:"Basic-Ant",
              robot_names[2]:None}
                
def robot_select_tab():
    options_menu = [sg.Text("Select robot: "), sg.OptionMenu(robot_names, robot_names[0], pad=(0,10), key="-ROBOT_SELECT-")]
    img = [sg.Image(filename="./docs/UI/UIdata/"+robot_pics[robot_names[0]], size=(800, 400), key="-ROBOT_IMAGE-")]

    main = sg.Column([[sg.VPush()], options_menu, img, [sg.VPush()]], expand_x=True, expand_y=True)

    tab = sg.Tab("Robot Select", [[main]])
    return tab;

agent_types = ["Full Random", "Sine Function Full", "Sine Function Half", "Step Cycle Half"]
agent_overview = {"Full Random":"Full Random agent\n    Starts off as a sequence of random actions for each motor for chosen amount of steps. Behavior of the agent is then made by repeating this sequence till end state is reached periodically.",
                  "Sine Function Full":"Sine Function Full agent\n    Each motor of the robot is controlled by sine wave. Values of these agents are made of only 4 parameters (amplitude, frequency, shiftX, shiftY) for each motor.",
                  "Sine Function Half":"Sine Function Full agent\n    Similar to Sine Function Full agent, however only half of robot's motors are controlled by sine waves. The other half is symmetrical (point symmetry through center of the body).",
                  "Step Cycle Half":"Step Cycle Half agent\n    Combination of random and half agent. STEPCOUNT long sequences of random actions for half of the motors are created and and then by symmetry transfered to opposing motors. During runtime, sequences of actions are repeatedly performed."}

def agent_select_tab():
    options_menu = [sg.Text("Select agent type: "), sg.OptionMenu(agent_types, agent_types[0], pad=(0,20,0,20), key="-AGENT_SELECT-")]
    frame = [sg.Frame("Agent overview", [[sg.Text(agent_overview[agent_types[0]], font=("Helvetica", 16), size=(72, None), pad=(20,20), key="-AGENT_OVERVIEW-")]], expand_x=True)]

    tab = sg.Tab("Agent config", [options_menu, frame])
    return tab;

def evolution_select_tab():
    tab = sg.Tab("Evolution config", [[]])
    return tab;

def make_window(theme=None):
    # layout_l = [[name('Text'), sg.Text('Text')],
    #             [name('Input'), sg.Input(s=15)],
    #             [name('Multiline'), sg.Multiline(s=(15,2))],
    #             [name('Output'), sg.Output(s=(15,2))],
    #             [name('Combo'), sg.Combo(sg.theme_list(), default_value=sg.theme(), s=(15,22), enable_events=True, readonly=True, k='-COMBO-')],
    #             [name('OptionMenu'), sg.OptionMenu(['OptionMenu',],s=(15,2))],
    #             [name('Checkbox'), sg.Checkbox('Checkbox')],
    #             [name('Radio'), sg.Radio('Radio', 1)],
    #             [name('Spin'), sg.Spin(['Spin',], s=(15,2))],
    #             [name('Button'), sg.Button('Button')],
    #             [name('ButtonMenu'), sg.ButtonMenu('ButtonMenu', sg.MENU_RIGHT_CLICK_EDITME_EXIT)],
    #             [name('Slider'), sg.Slider((0,10), orientation='h', s=(10,15))],
    #             [name('Listbox'), sg.Listbox(['Listbox', 'Listbox 2'], no_scrollbar=True,  s=(15,2))],
    #             [name('Image'), sg.Image(sg.EMOJI_BASE64_HAPPY_THUMBS_UP)],
    #             [name('Graph'), sg.Graph((125, 50), (0,0), (125,50), k='-GRAPH-')]  ]

    # layout_r  = [[name('Canvas'), sg.Canvas(background_color=sg.theme_button_color()[1], size=(125,50))],
    #             [name('ProgressBar'), sg.ProgressBar(100, orientation='h', s=(10,20), k='-PBAR-')],
    #             [name('Table'), sg.Table([[1,2,3], [4,5,6]], ['Col 1','Col 2','Col 3'], num_rows=2)],
    #             [name('Tree'), sg.Tree(treedata, ['Heading',], num_rows=3)],
    #             [name('Horizontal Separator'), sg.HSep()],
    #             [name('Vertical Separator'), sg.VSep()],
    #             [name('Frame'), sg.Frame('Frame', [[sg.T(s=15)]])],
    #             [name('Column'), sg.Column([[sg.T(s=15)]])],
    #             [name('Tab, TabGroup'), sg.TabGroup([[sg.Tab('Tab1',[[sg.T(s=(15,2))]]), sg.Tab('Tab2', [[]])]])],
    #             [name('Pane'), sg.Pane([sg.Col([[sg.T('Pane 1')]]), sg.Col([[sg.T('Pane 2')]])])],
    #             [name('Push'), sg.Push(), sg.T('Pushed over')],
    #             [name('VPush'), sg.VPush()],
    #             [name('Sizer'), sg.Sizer(1,1)],
    #             [name('StatusBar'), sg.StatusBar('StatusBar')],
    #             [name('Sizegrip'), sg.Sizegrip()]  ]

    # layout = [[sg.MenubarCustom([['File', ['Exit']], ['Edit', ['Edit Me', ]]],  k='-CUST MENUBAR-',p=0)] if use_custom_titlebar else [sg.Menu([['File', ['Exit']], ['Edit', ['Edit Me', ]]],  k='-CUST MENUBAR-',p=0)],
    #           [sg.Checkbox('Use Custom Titlebar & Menubar', use_custom_titlebar, enable_events=True, k='-USE CUSTOM TITLEBAR-')],
    #           [sg.T('PySimpleGUI Elements - Use Combo to Change Themes', font='_ 18', justification='c', expand_x=True)],
    #           [sg.Col(layout_l), sg.Col(layout_r)]]
    
    
    tabGroup = [[sg.TabGroup([[main_tab(), robot_select_tab(), agent_select_tab()]])]]

    window = sg.Window('Test GUI', tabGroup, font=font, finalize=True, keep_on_top=True)
    window['-ROBOT_SELECT-'].TKStringVar.trace("w", robot_select_callback)
    window['-AGENT_SELECT-'].TKStringVar.trace("w", agent_select_callback)

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
    window["-FRAME_TEXT-"].update(TEXT)

    window_values = {}

# Event Loop to process "events" and get the "values" of the inputs
    SETUP = True
    RUN = True
    while SETUP:
        event, values = window.read()
        print(event,values)
        # print(event)

        if event == sg.WIN_CLOSED or event == 'Exit':
            quit()
            break
        
        if event == "-ROBOT_SELECT-":
            window["-ROBOT_IMAGE-"].update(filename="./docs/UI/UIdata/"+robot_pics[values["-ROBOT_SELECT-"]], size=(800, 400))
            window.refresh()

        if event == "-AGENT_SELECT-":
            window["-AGENT_OVERVIEW-"].update(agent_overview[values["-AGENT_SELECT-"]])
            window.refresh()

        TEXT = '''- Robot selected: {}\n- Robot controlling agent: {}'''.format(values['-ROBOT_SELECT-'], values["-AGENT_SELECT-"])
        window["-FRAME_TEXT-"].update(TEXT)

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
