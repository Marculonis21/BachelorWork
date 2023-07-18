TUI module
==========

.. automodule:: TUI
   :members:
   :no-undoc-members:

List of possible arguments:
    * ``--experiment`` - takes one or more created experiment names (separated
      by white space),
    * ``--experiment_names`` - with this argument the program prints out a list
      of names of all created experiments,
    * ``--batch`` - integer input argument, specifies how many times is each
      selected experiment going to be repeated (all repetitions saved inside
      single batch folder)
    * ``--batch_note`` - optional note that can be added to the name of the batch
      folder, where batch results are going to be saved,
    * ``--no_graph`` - flag argument, when selected the evolutionary algorithm
      disables progress graph drawing,
    * ``--open`` - argument taking best individual file (``.save``) from previous
      experiment for visualization.

Usage examples:
    * Listing all created experiment names:

        .. code-block::

            >>> TUI.py --experiment_names
            <<< List of created experiments:
                 - exp10_TFS
                 - exp11_TFS_spot
                 - exp12_TFS_ant
                 ...

    * Running an experiment batch size of 5 (experiment is going to be repeated 5 times):

        .. code-block::

            >>> TUI.py --experiment exp11_TFS_spot --batch 5
            <<< Starting experiment - exp11_TFS_spot
                ...

    * Running multiple experiments all with batch size of 10 (each of selected
      experiments is going to be repeated 10 times) and we disable drawing of
      the progress graph. Each batch will be saved in it's own folder:

        .. code-block::

            >>> TUI.py --experiment exp11_TFS_spot exp12_TFS_ant --batch 10 --no_graph
            <<< Starting experiment - exp11_TFS_spot
                ...

    * Visualizing the best individual (loading best individual save file - ``.save`` file) of selected run from selected experiment:

        .. code-block::

            >>> TUI.py --open saved_files/runs/run1/individual.save
            <<< Ready for simulation - Press ENTER
            >>> <ENTER>
            <<< *Visualization starts*
                

