# Add a new module to the libs 
Four different types of modules can be implemented in the Automated Discovery Tool libs

- Systems
- Explorers
- Input_wrapper
- Output_representations

In order to implement a custom module,

1. Each module has its own folder.

    To add a new module create the file in the associated folder

    Example: 

        libs/adtool/adtool/explorers/myBeautifulNewExplorer.py

    or

        libs/adtool/adtool/systems/python_systems/myBeautifulNewPythonSystems.py

2. The new module must ultimately inherit from its module base class (BaseSystem, BaseOutputRepresentation...). 

    Please adhere to the parent class interface during the implementation.

    You can add decorators to your module class, which are useful to set module config parameters. These can be then directly set them in the GUI. 

    An example to implement a new explorer :

        from explorers import BaseExplorer
        from adtool.utils.config_parameters import StringConfigParameter

        @StringConfigParameter(name="a_string_parameter", possible_values=["first_possible_value", "second_possible_value"], default="first_possible_value")
        
        class NewExplorer(BaseExplorer):

            CONFIG_DEFINITION = {}

            def __init__(self, **kwargs) -> None:
                super().__init__(**kwargs)

            def initialize(self, input_space, output_space, input_distance_fn):
                super().initialize(input_space, output_space, input_distance_fn)
                """do some brilliant stuff"""
            
            def sample(self):
                """do some brilliant stuff"""

            def observe(self, parameters, observations):
                """do some brilliant stuff"""

            def optimize(self):
                """do some brilliant stuff"""

    Don't forget kwargs argument in the `__init__` method and `CONFIG_DEFINITION` class variable.

3. Add import in `libs/adtool/module_cat/subfolder_if_needed/__init__.py`

    Example: 

        libs/adtool/adtool/explorers/__init__.py
        libs/adtool/adtool/systems/python_systems/__init__.py

4. Add new module in `registration.py` in `REGISTRATION` dict

    Modify REGISTRATION starting from this

        REGISTRATION = {
                'systems':{
                    'PythonLenia': PythonLenia,
                },...
            }

    to for example the following

        REGISTRATION = {
            'systems':{
                'PythonLenia': PythonLenia,
                'MyBeautifullNewPythonSystems': MyBeautifullNewPythonSystems,
            },...
        }

    The dict keys are used in GUI to choose your module when you setup an experiment.
