# Add a new callback to the libs 
Six different types of callbacks can be implemented in the `Automated Discovery Tool` libs, namely:
- on_cancelled_callbacks : called when user decides to cancel the experiment in the frontend
- on_discovery_callbacks : called each time the experiment makes a new discovery
- on_error_callbacks : called when experiment raises an error during execution
- on_finished_callbacks : called when the experiment ends
- on_save_callbacks : called when the experiment saves its state
- on_save_finished_callbacks : called when the experiment has finished saving


1. Each callback type has its own folder. 
To add a new callback create the file in the associated folder

    Example: 

    ```
    libs/adtool/adtool/utils/callbacks/on_cancelled_callbacks/my_beautiful_new_on_cancelled_callback.py
    ```
    or
    ```
    libs/adtool/adtool/utils/callbacks/on_discovery_callbacks/my_beautiful_new_on_discovery_callback.py
    ```

2. The new callback must inherits base callback class of its own type.

   Example:

   ```
        class MyBeautifulNewOnCancelledCallback(BaseOnCancelledCallback):
            pass
   ```
    or
   ```
        class MyBeautifulNewOnDiscoveryCallback(BaseOnDiscoveryCallback):
            pass
   ```
    A full example:
    ```
        from utils.callbacks.on_discovery_callbacks import BaseOnDiscoveryCallback
        
        class MyBeautifullNewOnDiscoveryCallback(BaseOnDiscoveryCallback):

            def __init__(self, folder_path, to_save_outputs, **kwargs) -> None:
                super().__init__(to_save_outputs, **kwargs)
                """do some brilliant stuff"""

            def __call__(self, **kwargs) -> None:
                """do some brilliant stuff"""
    ```
    
    Don't forget to pass `**kwargs` in the `__init__` method.
    Each time our callback will be raised the `__call__` method will be executed.


3. Add import header file, e.g., `libs/adtool/adtool/utils/callbacks/callbacks_sub_folder/__init__.py`

    Example: 

    ```
    libs/adtool/adtool/utils/callbacks/on_discovery_callbacks/__init__.py
    libs/adtool/adtool/utils/callbacks/on_cancelled_callbacks/__init__.py
    ```
4. Add new callback in registration.py in REGISTRATION dict

   Example modification:

   ```
   Modify REGISTRATION like this:

   REGISTRATION = {
        ...
        'callbacks': {
            'on_discovery':{
                'base': on_discovery_callbacks.BaseOnDiscoveryCallback,
                'disk': on_discovery_callbacks.OnDiscoverySaveCallbackOnDisk
            },
            'on_cancelled':{
                'base': on_cancelled_callbacks.BaseOnCancelledCallback
            },
            'on_error':{
                'base': on_error_callbacks.BaseOnErrorCallback
            },
            'on_finished':{
                'base': on_finished_callbacks.BaseOnFinishedCallback
            },
            'on_saved':{
                'base': on_save_callbacks.BaseOnSaveCallback,
                'disk': on_save_callbacks.OnSaveModulesOnDiskCallback
            },
            'on_save_finished':{
                'base': on_save_finished_callbacks.BaseOnSaveFinishedCallback
            },
        },...
    }

    ==>
    
    REGISTRATION = {
        ...
        'callbacks': {
            'on_discovery':{
                'base': on_discovery_callbacks.BaseOnDiscoveryCallback,
                'disk': on_discovery_callbacks.OnDiscoverySaveCallbackOnDisk
                'new': MyBeautifullNewOnDiscoveryCallback
            },
            'on_cancelled':{
                'base': on_cancelled_callbacks.BaseOnCancelledCallback
            },
            'on_error':{
                'base': on_error_callbacks.BaseOnErrorCallback
            },
            'on_finished':{
                'base': on_finished_callbacks.BaseOnFinishedCallback
            },
            'on_saved':{
                'base': on_save_callbacks.BaseOnSaveCallback,
                'disk': on_save_callbacks.OnSaveModulesOnDiskCallback
            },
            'on_save_finished':{
                'base': on_save_finished_callbacks.BaseOnSaveFinishedCallback
            },
        },...
    }
   ```

5. For now the software does not permit to add custom callback via GUI. You may use the libs independently like in `libs/test/AutoDiscExperiment.py` and add manually your personal callbacks. The other way is to manually add your callback in `services/AutodiscServer/flask/experiments/remote_experiments.py` on the `__init__` method to use your own callbacks with the full software package.
