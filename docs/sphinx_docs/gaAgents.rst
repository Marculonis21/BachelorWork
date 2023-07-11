gaAgents module
===============

.. automodule:: gaAgents
   :no-members:

.. autoenum:: gaAgents.EvoType
   :members:

.. autoclass:: gaAgents.BaseAgent
   :no-undoc-members:

.. _decorators:
Genetic operator decorator
~~~~~~~~~~~~~~~~~~~~~~~~~~

Genetic operators (:func:`BaseAgent.selection`, :func:`BaseAgent.crossover`,
:func:`BaseAgent.mutation`) are using special custom decorators to enable GUI
configuration. 

When :attr:`gui` flag is enabled on init, which shows that agent was created
through GUI, these decorators switch from using operator methods provided in
code to using the ones in `genetic_operators` dictionary (dictionary of
delegates to genetic operators for selection, crossover, mutation).

Example of decorator::

    def selection_deco(func): # get current function

        def wrapper(self, population, fitness_values): # define wrapper
            # if GUI is selected - use delegate from `genetic_operators`
            if self.gui:                
                method, arguments = self.genetic_operators["selection"] 
                return method(population, fitness_values, *arguments)

            # otherwise use default function (from code)
            else:
                return func(self, population, fitness_values)

        return wrapper
