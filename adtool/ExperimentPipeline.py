import traceback
from copy import deepcopy
from typing import Callable, List

from adtool.utils.leaf.Leaf import Leaf, prune_state
from adtool.utils.leaf.locators.locators import BlobLocator

from os import listdir
import os
from os.path import isfile, join
import json

import torch


def replace_lists_with_tensor(d):
    # if we found a list of floats, convert it to a tensor, then if we found list of tensors, convert it to a tensor etc from bottom-up
    if isinstance(d, list) and all(isinstance(i, float) for i in d):
        return torch.tensor(d).squeeze()
    elif isinstance(d, list):
        return [replace_lists_with_tensor(i) for i in d]
    elif isinstance(d, dict):
        return {k:replace_lists_with_tensor(v) for k,v in d.items()}
    else:
        return d


def replace_lists_with_numpy(d):
    # if we found a list of floats, convert it to a tensor, then if we found list of tensors, convert it to a tensor etc from bottom-up
    if isinstance(d, list) and all(isinstance(i, float) for i in d):
        return torch.tensor(d).squeeze().numpy()
    elif isinstance(d, list):
        return [replace_lists_with_numpy(i) for i in d]
    elif isinstance(d, dict):
        return {k:replace_lists_with_numpy(v) for k,v in d.items()}
    else:
        return d

class ExperimentPipeline(Leaf):
    """
    Pipeline of an automated discovery experiment.

    An experiment is at least constituted of a system and an explorer.

    Additionally, input wrappers and output representations can be added and
    composed.

    In order to monitor the experiment, you must provide **callbacks**, which
    will be called every time a discovery has been made.
    Please see: `callbacks.base_callback.BaseCallback`.
    """

    def __init__(
        self,
        config: dict={},
        experiment_id: int = 0,
        seed: int = 0,
        system=None,
        explorer=None,
        save_frequency: int = 100,
        on_discovery_callbacks: List[Callable] = [],
        on_save_finished_callbacks: List[Callable] = [],
        on_finished_callbacks: List[Callable] = [],
        on_cancelled_callbacks: List[Callable] = [],
        on_save_callbacks: List[Callable] = [],
        on_error_callbacks: List[Callable] = [],
        logger=None,
        resource_uri: str = "",
        discovery_saving_keys: List[str] = [],
    ) -> None:
        """
        Initializes state of experiment pipeline, setting all necessary
        attributes given by the following arguments.

        #### Args:
        - **experiment_id**: ID of current experiment
        - **seed**: Current seed for random number generation
        - **system**: System to be explored
        - **explorer**: Explorer used
        - **save_frequency**: Frequency to save state of the experiment
        - **on_discovery_callbacks**: Called when a discovery is made
        - **on_save_finished_callbacks**: Called when save is complete
        - **on_finished_callbacks**: Called when the experiment is finished
        - **on_cancelled_callbacks**: Called when the experiment is cancelled
        - **on_save_callbacks**: Called when a experiment save is made
        - **on_error_callbacks**: Called when an error is raised
        """
        super().__init__()
        self.config = config
        self.locator = BlobLocator()
        self.locator.resource_uri = resource_uri

        # METADATA
        self.run_idx = 0
        self.experiment_id = experiment_id
        self.seed = seed
        self.save_frequency = save_frequency
        self.logger = logger

        # stores the original resource_uri from the config
        # we pass this to the configs, because self.locator.resource_uri
        # gets overridden frequently by the save routine
        self.resource_uri = resource_uri

        self.discovery_saving_keys = discovery_saving_keys

        # SYSTEM

        self._system = system

        # EXPLORER
        self._explorer = explorer

        # CALLBACKS
        self._on_discovery_callbacks = on_discovery_callbacks
        self._on_save_finished_callbacks = on_save_finished_callbacks
        self._on_finished_callbacks = on_finished_callbacks
        self._on_cancelled_callbacks = on_cancelled_callbacks
        self._on_error_callbacks = on_error_callbacks
        self._on_save_callbacks = on_save_callbacks

    def _raise_callbacks(self, callbacks: List[Callable], **kwargs) -> None:
        """
        Raise all callbacks linked to the current event
        (new discovery, a save, the end of the experiment...)

        Args:
            callbacks: list of all callbacks must be raise
            **kwargs: e.g., self.run_idx, seed, experiment_id, some modules...
        """
        for callback in callbacks:
            callback(**kwargs)

    def run(self, n_exploration_runs: int):
        """
        Launches the experiment for `n_exploration_runs` explorations.

        `n_exploration_runs` is specified so more optimized looping routines
        can be chosen in the inner loop if desired. Interfacing with the
        objects in the inner loop should be done via the callback interface.

        #### Args:
        - **n_exploration_runs**: number of explorations

        #### Returns:
        - **LeafUID**: returns the UID associated to the experiment
        """
        try:
            data_dict = self._explorer.bootstrap(
            )

            

            mypath = self.config['experiment']['config']['save_location']+"discoveries/"
            #list all discovery.json files in subdirectories  
            #check if mypath exists and is a folder
            if not os.path.exists(mypath):
                os.makedirs(mypath)

            discoveries_folders = [f for f in listdir(mypath) if not isfile(join(mypath, f))]
            #get discovery.json files in discoveries folders
            # log a warning message if not empty
            self.logger.warning(
                f"[PRELOAD] - Loading previous discoveries from experiment"
            )

            json_discoveries = [ 
                folder for folder in discoveries_folders for f in listdir(join(mypath, folder))
                if isfile(join(mypath, folder, f)) and
                   f=="discovery.json" ]
            
            for json_discovery in json_discoveries:
                # check if config.json is the same
                
                # with open(join(mypath, json_discovery,"config.json")) as f:
                #     discovery_config=json.load(f)
                #     if discovery_config!=self.config:
                #         #TODO: check que les entrées sorties + warning sur les différences
                #         raise Exception("The discovery config is not the same as the current config")
                with open(join(mypath, json_discovery,"discovery.json")) as f:
                    new_trial_data = json.load(f)
                    #replace each list of list of floats with a tensor, recursively but bottom-up

                    new_trial_data = replace_lists_with_numpy(new_trial_data)



                  #  print(new_trial_data)

                    
                    self._explorer._history_saver.map( new_trial_data )
            self.logger.info(
                "[LOADED] - Loaded previous discoveries from experiment"
                f"{self.experiment_id} with seed {self.seed}"
            )
            




            while self.run_idx < n_exploration_runs:
                # check  if target.json exists
                if os.path.exists(f"{mypath}/target.json"):
                    with open(f"{mypath}/target.json") as f:
                        target=json.load(f)
                        #replace each list of list of floats with a tensor, recursively but bottom-up
                        # target = replace_lists_with_tensor(target)
                        data_dict['target']=target['target']


                # pass trial parameters through system
                data_dict = self._system.map(data_dict)

                # render system output
                rendered_output,ext = self._system.render(data_dict)

                # exploration phase : emits new trial parameters for next loop
                data_dict = self._explorer.map(data_dict)    
        

                discovery = self._explorer.read_last_discovery()


                # pass new dict to prevent in-place mutation by callback
                discovery_to_save = deepcopy(discovery)

                # use discovery_saving_keys as a mask for what to save
                if len(self.discovery_saving_keys) > 0:
                    for key in list(discovery_to_save.keys()):
                        if key not in self.discovery_saving_keys:
                            del discovery_to_save[key]

                # TODO: pass the rendered output more easily
            #    discovery_to_save["rendered_output"] = rendered_output

                self._raise_callbacks(
                    self._on_discovery_callbacks,
                    config=self.config,
                    resource_uri=self.resource_uri,
                    run_idx=self.run_idx,
                    experiment_id=self.experiment_id,
                    seed=self.seed,
                    discovery=discovery_to_save,
                    rendered_output=rendered_output,
                    rendered_output_extension=ext,

                    # run_parameters=discovery[self._explorer.postmap_key],
                    # output=discovery[self._explorer.premap_key],
                    # raw_output=discovery["raw_" + self._explorer.premap_key],
                    # binaries={"rendered_output": rendered_output}
                )

                self.logger.info(
                    "[DISCOVERY] - New discovery from experiment"
                    f"{self.experiment_id} with seed {self.seed}"
                )

                # avoids divide by zero
                run_idx_start_from_one = self.run_idx + 1

                if (
                    run_idx_start_from_one % self.save_frequency == 0
                    or run_idx_start_from_one == n_exploration_runs
                ):
                    self.save(resource_uri=self.resource_uri)


                self.run_idx += 1

        except Exception:
            message = "error in experiment {} self.run_idx {} seed {} = {}".format(
                self.experiment_id, self.run_idx, self.seed, traceback.format_exc()
            )

            # TODO: do this in appdb side
            if len(message) > 8000:  # Cut to match varchar length of AppDB
                message = message[:7997] + "..."

            self.logger.error("[ERROR] - " + message)
            self._raise_callbacks(
                self._on_error_callbacks,
                run_idx=self.run_idx,
                seed=self.seed,
                experiment_id=self.experiment_id,
            )
            raise

        # CLEANUP


        self.logger.info(
            "[FINISHED] - experiment {} with seed {} finished".format(
                self.experiment_id, self.seed
            )
        )

        self._raise_callbacks(
            self._on_finished_callbacks,
            run_idx=self.run_idx,
            seed=self.seed,
            experiment_id=self.experiment_id,
        )
        return

    def save(self, resource_uri: str):
        self._raise_callbacks(
            self._on_save_callbacks,
            experiment_id=self.experiment_id,
            seed=self.seed,
            module_to_save=self,
            resource_uri=resource_uri,
        )

        # only run on_save_finished callbacks if on_save_callbacks
        # provide a uid for the save
        if self.__dict__.get("uid", None) is not None:
            self._raise_callbacks(
                self._on_save_finished_callbacks,
                uid=self.uid,
                report_dir=resource_uri,
                experiment_id=self.experiment_id,
                seed=self.seed,
                run_idx=self.run_idx,
            )

            # uid is passed by modifying the parent object, to enable
            # communication between the before and after callbacks
            # so restore its original state after all the callbacks
            del self.uid

        self.logger.info(
            "[SAVED] - experiment {} with seed {} saved".format(
                self.experiment_id, self.seed
            )
        )
        return

    @prune_state(
        {
            "_on_discovery_callbacks": [],
            "_on_save_finished_callbacks": [],
            "_on_finished_callbacks": [],
            "_on_cancelled_callbacks": [],
            "_on_error_callbacks": [],
            "_on_save_callbacks": [],
        }
    )
    def serialize(self) -> bytes:
        return super().serialize()
