#!/usr/bin/env python

import roboEvo

import resources.GUI_tabs.main_tab as main_tab
import resources.GUI_tabs.robot_tab as robot_tab
import resources.GUI_tabs.agent_tab as agent_tab 
import resources.GUI_tabs.run_window as run_window
import resources.GUI_tabs.evo_tab as evo_tab

import PySimpleGUI as sg
import threading

font = ("Helvetica", 15)

def make_window():
    tabGroup = [[sg.TabGroup([[main_tab.tab(), robot_tab.tab(), agent_tab.tab(), evo_tab.tab()]], size=(800,600))]]

    window = sg.Window('Test GUI', tabGroup, size=(800,600), font=font, finalize=True,  use_default_focus=False)

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
        main_tab.events(window, event, values, robot_tab=robot_tab, agent_tab=agent_tab)

        main_tab.set_overview_text(window, values, robot_tab, agent_tab)

        if event == "-START-":
            window_values = values
            window.close()
            break
        
    # Starting experiment window
    window = run_window.window()
            
    def start_run():
        roboEvo.run_experiment(run_window.get_params(window_values, robot_tab, agent_tab), gui=True)

    working_thread = threading.Thread(target=start_run, daemon=True)
    working_thread.start()

    run_window.draw_chart(window);

    while True:
        if working_thread.is_alive():
            event, values = window.read(timeout=2500)
        else:
            window["-EXIT-"].update(button_color="green")
            event, values = window.read()
            break

        if event == "-RUN_PREVIEW-":
            roboEvo.raise_preview()

        if (event == sg.WIN_CLOSED or event == 'Exit') or event == "-EXIT-":
            roboEvo.raise_abort()
            working_thread.join()
            break

        run_window.update_chart(window)

        window["-GENNUM-"].update(str(roboEvo.GUI_GEN_NUMBER))
        if len(roboEvo.GUI_FITNESS) > 0:
            window["-MEANFIT-"].update(str(roboEvo.GUI_FITNESS[0]))
            window["-MINFIT-"].update(str(roboEvo.GUI_FITNESS[1]))
            window["-MAXFIT-"].update(str(roboEvo.GUI_FITNESS[2]))

    window.close()
