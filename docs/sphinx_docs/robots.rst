robots module
=============

.. automodule:: robots
   :no-members:

.. autoclass:: robots.BaseRobot
   :no-undoc-members:

.. _body_parts_symbols:
Body parts symbols
~~~~~~~~~~~~~~~~~~
    Robot's changable body parts are configured inside its XML config file.

    We can use two types of symbols:
        * ``$...$`` - creates variable with name and default value inside XML
          file which can be found by our parser and changed during algorithm
          runtime.
        * ``@...@`` - creates a field which can have arithmetic operations that
          are going to be evaluated after all ``$...$`` variables were
          assigned their values.

Variable symbols
""""""""""""""""
    Variable symbol have specified format which they must use to be correctly
    recognized (parses uses exhaustive regular expressions to find symbols).
    
    The correct format is as follows ``$<variable_name>(<default_length>)$``, where
        #. <variable_name> - have to include at least one character and it can include lower and upper case letters, numbers and underscore ('_') (= characters from group [A-Za-z0-9_]).
        #. <default_length> - have to include a value - positive/negative float
    The variable's name inside our program will use the specified name from the
    XML file. 

    A single variable can be used multiple times inside the XML file. Then it
    must have the same format at all places, but only the first encounter of
    the symbol will create actual variable inside our program. The other ones
    are used only during value assignment.

Arithmetic symbols
""""""""""""""""""
    Arithmetic symbols are similarly to variables searched for by regular
    expressions. For the possible scope of uses the definition is really weak
    so the correctness of operations used inside the symbol is up to the user
    to check.

    The arithmetic expression inside the symbol is evaluated with python's
    ``eval()`` method (`Eval documentation <https://docs.python.org/3/library/functions.html#eval>`_)
    and the evaluation always occurs after all defined variables were assigned
    (even if they are not used inside any expression).

    Usage examples:
        #. ``@10+2@`` - in this case text inside XML file at this place will be evaluated to 12 (basic and not really useful example of usage).
        #. ``@-1.8-$THIGH_Z(-0.8)$@`` - in this case, first the variable :attr:`THIGH_Z` is assigned its value (let's say default -0.8) and then the arithmetic expression is evaluated with result -1 (example from *spot_like.xml* config file)





