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
    frame_layout = [[sg.Text("", font=("Helvetica", 16), size=(100, 20), expand_x=True, key="-FRAME_TEXT-")]]

    frame = [[sg.Frame("Settings overview", frame_layout, size=(800, 400))]]

    options_l = [[sg.Checkbox("Show final run", key="-CB_FINAL-")],
                 [sg.Checkbox("Show progress runs", key="-CB_PROGRESS-")],
                 [sg.Sizer(0, 1)]]

    options_r = [[sg.Checkbox("Show graph", key="-CB_GRAPH-")],
                 [sg.Checkbox("Save best agent", key="-CB_SAVEBEST-")],
                 [sg.Text("Save directory:"), sg.Text(".", key="-SAVE_DIR-")],
                 [sg.FolderBrowse("Browse", target="-SAVE_DIR-")]]

    options = [[sg.Col([[sg.Col(options_l, element_justification='l', vertical_alignment='t', expand_x=True), sg.Col(options_r, element_justification='l', vertical_alignment='t', expand_x=True)]], expand_x=True)]]

    start = [[sg.Col([[sg.Button("Start", key="-START-")]], expand_x=True, element_justification='r')]]

    main = frame + options + start

    tab = sg.Tab("Main", main)
    return tab

def robot_select_callback(var, index, mode):
    window.write_event_value('-ROBOT_SELECT-', window['-ROBOT_SELECT-'].TKStringVar.get())

def robotselection_tab():
    options_menu = [sg.Text("Select robot: "), sg.OptionMenu(["OpenAi Ant", "Basic Ant"], "OpenAi Ant", pad=(0,10), key="-ROBOT_SELECT-")]
    img = [sg.Image(filename="./docs/UI/UIdata/Ant-v3", size=(800, 400), key="-ROBOT_IMAGE-")]

    main = sg.Column([[sg.VPush()], options_menu, img, [sg.VPush()]], expand_x=True, expand_y=True)

    tab = sg.Tab("Robot Select", [[main]])
    return tab;

def algconfig_tab():
    tab = sg.Tab("Algorithm config", [[]])
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
    
    
    tabGroup = [[sg.TabGroup([[main_tab(), robotselection_tab()]])]]

    window = sg.Window('Test GUI', tabGroup, font=font, finalize=True, keep_on_top=True)
    window['-ROBOT_SELECT-'].TKStringVar.trace("w", robot_select_callback)

    return window

def make_run_window():
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
    
    layout = [[sg.Canvas(key='-FIG-')],
              [sg.Text("Gen Number: "), sg.Text("", key="-GENNUM-")],
              [sg.Text("Max Fit: "), sg.Text("", key="-MAXFIT-")], [sg.Text("Min Fit: "), sg.Text("", key="-MINFIT-")], [sg.Text("Mean Fit: "), sg.Text("", key="-MEANFIT-")]]

    window = sg.Window('Process running', layout, font=font, finalize=True, keep_on_top=True)

    return window

if __name__ == "__main__":
# Create the Window
    window = make_window()
    _, values = window.read(timeout=0)

    TEXT = '''
    - Robot selected: {}
    - Robot controlling agent: {}
    '''.format(values['-ROBOT_SELECT-'], "Random agent")

    window["-FRAME_TEXT-"].update(TEXT)

# Event Loop to process "events" and get the "values" of the inputs
    SETUP = True
    RUN = True
    while SETUP:
        event, values = window.read(timeout=1000)
        # print(event)

        if event == sg.WIN_CLOSED or event == 'Exit':
            break
        
        if event == "-ROBOT_SELECT-":
            value = values["-ROBOT_SELECT-"]
            if value == "Basic Ant":
                value = "Basic-Ant"
            else:
                value = "Ant-v3"

            window["-ROBOT_IMAGE-"].update(filename="./docs/UI/UIdata/"+value,
                                        size=(800, 400))
            window.refresh()

        if event == "-START-":
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

    def startRun():
        antGA.RunFromGui(agent='random', show_best=True)

    threading.Thread(target=startRun, daemon=True).start()
    drawChart();

    while RUN:
        event, values = window.read(timeout=2000)
        print(event)

        if event == sg.WIN_CLOSED or event == 'Exit':
            break

        updateChart()

        window["-GENNUM-"].update("{}".format(antGA.GUI_GEN_NUMBER))
        if len(antGA.GUI_FITNESS) > 0:
            window["-MEANFIT-"].update("{}".format(antGA.GUI_FITNESS[0]))
            window["-MINFIT-"].update("{}".format(antGA.GUI_FITNESS[1]))
            window["-MAXFIT-"].update("{}".format(antGA.GUI_FITNESS[2]))

    window.close()
