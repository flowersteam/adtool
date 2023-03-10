from typing import Callable
from addict import Dict
from auto_disc.utils.logger import AutoDiscLogger


class BaseAutoDiscModule:
    '''
    Base class of all modules usable in auto_disc.
    '''
    _access_history = None  # Function to access (readonly) history of (input, output) pairs. Takes 1 positional argument which could be an index or a slice.
    # Function to ask history of outputs to be updated (use this if some output_representations changed)
    _call_output_history_update = None
    # Function to ask history of run_parameters to be updated (use this if some input_wrappers changed)
    _call_run_parameters_history_update = None
    CURRENT_RUN_INDEX = 0
    """
    Integer tracking index of current loop iteration
    """

    def __init__(self, logger: AutoDiscLogger = None, **kwargs) -> None:
        """
            Initialize attributes common to all auto_disc modules

            #### Args:
            - logger: the logger which will make it possible to keep information on the progress of an experiment in the database or on files
        """
        self.logger = logger

    def set_history_access_fn(self, function: Callable) -> None:
        '''
            Set the function allowing module to access (readonly) its history of (input, output) pairs.

            #### Args:
            - function: The method used to acces to the module history
        '''
        self._access_history = function

    def set_call_output_history_update_fn(self, function: Callable) -> None:
        '''
            Set the function asking a refresh (raw_outputs will be processed again with output representations) of all outputs in history.

            #### Args:
            - function: The method used to refresh
        '''
        self._call_output_history_update = function

    # def set_call_run_parameters_history_update_fn(self, function):
    #     '''
    #         Set the function asking a refresh (raw_run_parameters will be processed again with output representations) of all run_parameters in history.
    #     '''
    #     self._call_run_parameters_history_update = function

    def save(self) -> Dict:
        '''
            Return a dict storing everything that has to be pickled.

            **NOTE: Saving/loading is currently highly unstable, and the interface will be deprecated
            soon during an API update.**
        '''
        return {}

    def load(self, saved_dict: Dict):
        '''
            Reload module given a dict storing everything that had to be pickled.

            **NOTE: Saving/loading is currently highly unstable, and the interface will be deprecated
            soon during an API update.**

            #### Args:
            - saved_dict: The dict used to restore the module
        '''
        pass
