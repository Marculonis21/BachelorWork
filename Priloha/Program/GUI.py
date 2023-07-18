#!/usr/bin/env python
"""GUI module

Central GUI module. GUI splits into different tabs. GUI is created with
**PySimpleGUI** library.

GUI firstly initialises window with :func:`make_window` which collects windows
(created tabs) from all created tabs and puts them inside one big tabgroup.
More about those tabs inside individual sub-modules.

User should have some knowledge of **PySimpleGUI** library to fully understand
the GUI app. It is not in our capabilities to fully describe every part of
this library in exhaustive detail (more about this library in the official
`documentation <https://www.pysimplegui.org/en/latest/>`_).

After initialisation the app runs in infinite loop, reading inputs as events,
which are handled inside corresponding tabs.

Each tab has to have 2 mandatory functions: 
    * :func:`tab` - for initialising own tab
    * :func:`events` - for receiving ALL events and deciding if the event should be handled inside that tab
"""

import PySimpleGUI as sg
import threading
import numpy as np

import roboEvo

import resources.GUI_tabs.main_tab as main_tab
import resources.GUI_tabs.robot_tab as robot_tab
import resources.GUI_tabs.agent_tab as agent_tab 
import resources.GUI_tabs.run_window as run_window
import resources.GUI_tabs.evo_tab as evo_tab

DEFAULT_FONT = ("Arial", 15)

def make_window():
    """
    Main function initialising all other tabs and putting everything together
    inside single big ``TabGroup``. With this element the main app window is
    created and returned to the main infinite event loop.
    """

    tabGroup = [[sg.TabGroup([[main_tab.tab(), robot_tab.tab(), agent_tab.tab(), evo_tab.tab()]], size=(800,600))]]

    window = sg.Window('RoboEvo GUI', tabGroup, size=(800,600), font=DEFAULT_FONT, finalize=True,  use_default_focus=False)

    window['-AGENT_OVERVIEW_MORE-'].block_focus()
    window['-ROBOT_OVERVIEW_MORE-'].block_focus()
    window['-AGENT_SELECT-'].block_focus()
    window['-ROBOT_SELECT-'].block_focus()

    return window

if __name__ == "__main__":
    # Create the Window
    window = make_window()
    _, values = window.read(timeout=0)

    agent_tab.set_agent(agent_tab.agent_names[0], window)
    robot_tab.set_robot(robot_tab.robot_names[0], window)
    main_tab.set_overview_text(window, values, robot_tab, agent_tab)

    window_values = {}

    # Event Loop to process "events" and get the "values" of the inputs
    while True:
        event, values = window.read()

        if event == sg.WIN_CLOSED or event == 'Exit':
            quit()

        robot_tab.events(window, event, values, agents=agent_tab.agents)
        agent_tab.events(window, event, values)
        evo_tab.events(window, event, values)
        main_tab.events(window, event, values, robot_tab, agent_tab, evo_tab)

        main_tab.set_overview_text(window, values, robot_tab, agent_tab)

        if event == "-START-":
            window_values = values
            window.close()
            break
        
    # Starting experiment window
    window = run_window.window()
            
    params = run_window.get_params(window_values, robot_tab, agent_tab)
    def start_run():
        roboEvo.run_experiment(params, gui=True)

    working_thread = threading.Thread(target=start_run, daemon=True)
    working_thread.start()

    run_window.draw_chart(window);

    while True:
        if working_thread.is_alive():
            event, values = window.read(timeout=2500)
        else:
            window["-EXIT-"].update(button_color="green")
            window["-RUN_PREVIEW-"].update(disabled=True)
            event, values = window.read()
            break

        if event == "-RUN_PREVIEW-":
            roboEvo.GUI_PREVIEW = True

        if (event == sg.WIN_CLOSED or event == 'Exit') or event == "-EXIT-":
            roboEvo.GUI_ABORT = True
            working_thread.join()
            break

        run_window.update_chart(window)

        window["-RUN_PREVIEW-"].update(disabled=isinstance(params.agent, roboEvo.gaAgents.NEATAgent))

        window["-GENNUM-"].update(str(roboEvo.GUI_GEN_NUMBER))
        if isinstance(roboEvo.EPISODE_HISTORY, dict):
            window["-MEANFIT-"].update("NEAT - disabled")
            window["-MINFIT-"].update("NEAT - disabled")
            window["-MAXFIT-"].update("NEAT - disabled")
        elif len(roboEvo.EPISODE_HISTORY) > 0:
            mean = np.mean(roboEvo.EPISODE_HISTORY[-1])
            min  = np.min(roboEvo.EPISODE_HISTORY[-1])
            max  = np.max(roboEvo.EPISODE_HISTORY[-1])
            window["-MEANFIT-"].update(mean)
            window["-MINFIT-"].update(min)
            window["-MAXFIT-"].update(max)

    window.close()
