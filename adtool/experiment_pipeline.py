import traceback
import typing
from copy import copy
from typing import Any, Callable, Dict, List, Type

import torch
from adtool import BaseAutoDiscModule
from adtool.input_wrappers import BaseInputWrapper
from adtool.input_wrappers.generic import DummyInputWrapper
from adtool.auto_disc.output_representations import BaseOutputRepresentation
from adtool.auto_disc.output_representations.generic import DummyOutputRepresentation
from adtool.utils.callbacks.interact_callbacks import Interact
from adtool.utils.misc import DB

from adtool.auto_disc import explorers

from adtool.auto_disc import systems

class CancellationToken:
    """
    Manages the cancellation token which allows you to stop an experiment in progress
    """

    def __init__(self):
        """
        Init the cancellation token to false
        """
        self._token = False

    def get(self) -> bool:
        """
        Give acces to the cancellation token

        Returns:
            _token: a boolean indicating if the current experiment must be cancelled
        """
        return self._token

    def trigger(self):
        """
        Set the cancellation token to true (the experiment must be cancelled)
        """
        self._token = True


class ExperimentPipeline:
    """
    Pipeline of an automated discovery experiment.
    An experiment is at least constitued of a system and an explorer. Additionally, input wrappers and output representations can be added (multiple can be stacked).
    When the system requires an action at each timestep, an `action_policy` must be provided.
    In order to monitor the experiment, you must provide `on_exploration_classbacks`, which will be called every time a discovery has been made. Please provide callbacks overriding the `libs.auto_disc.utils.BaseAutoDiscCallback`.
    """

    def __init__(
        self,
        experiment_id: int,
        seed: int,
        system: systems,
        explorer: explorers,
        input_wrappers: List[BaseInputWrapper] = None,
        output_representations: typing.List[BaseOutputRepresentation] = None,
        action_policy=None,
        save_frequency: int = 100,
        on_discovery_callbacks: typing.List[Callable] = [],
        on_save_finished_callbacks: typing.List[Callable] = [],
        on_finished_callbacks: typing.List[Callable] = [],
        on_cancelled_callbacks: typing.List[Callable] = [],
        on_save_callbacks: typing.List[Callable] = [],
        on_error_callbacks: typing.List[Callable] = [],
        interact_callbacks: typing.List[Callable] = [],
    ) -> None:
        """
        Init experiment pipeline. Set all attribute used by the pipeline.

        Args:
            experiment_id: Id of current experiment
            seed: Current seed number
            system: The system we want explore in this experiment
            explorer: The explorer we use to explore the system
            input_wrappers: A list of wrapper who redine the input spaces
            output_representations: A list of wrapper who redine the output spaces
            action_policy: A method that indicate what action should be taken based on the policy
            save_frequency: defines the frequency with which the backups of the elements of the experience are made
            on_discovery_callbacks: Callbacks raised when a discovery was made
            on_save_finished_callbacks: Callbacks raised when a the experiment save is complete
            on_finished_callbacks: Callbacks raised when the experiment is over
            on_cancelled_callbacks: Callbacks raised when the experiment is canccelled
            on_save_callbacks: Callbacks raised when a experiment save was made
            on_error_callbacks: Callbacks raised when an error is raised

        """
        self.experiment_id = experiment_id
        self.seed = seed
        self.save_frequency = save_frequency

        self.db = DB()

        def access_history_fn(
            keys: typing.List[str] = [], new_keys=["idx", "input", "output"]
        ) -> Callable:
            return lambda index=slice(None, None, None): self.db.to_autodisc_history(
                self.db[index], keys, new_keys
            )

        ### SYSTEM ###
        self._system = system
        self._system.set_call_output_history_update_fn(self._update_outputs_history)
        # self._system.set_call_run_parameters_history_update_fn(self._update_run_parameters_history)
        self._system.set_history_access_fn(
            access_history_fn(
                keys=["idx", "run_parameters", "raw_output"],
                new_keys=["idx", "input", "output"],
            )
        )

        ### OUTPUT REPRESENTATIONS ###
        if output_representations is not None and len(output_representations) > 0:
            self._output_representations = output_representations
        else:
            self._output_representations = [DummyOutputRepresentation()]

        for i in range(len(self._output_representations)):
            self._output_representations[i].set_call_output_history_update_fn(
                lambda: self._update_outputs_history(i)
            )
            if i == 0:
                self._output_representations[i].initialize(
                    input_space=self._system.output_space
                )
                input_key = "raw_output"
            else:
                input_key = f"output_{i-1}"
                self._output_representations[i].initialize(
                    input_space=self._output_representations[i - 1].output_space
                )

            if i == len(self._output_representations) - 1:
                output_key = f"output"
            else:
                output_key = f"output_{i}"

            self._output_representations[i].set_history_access_fn(
                access_history_fn(
                    keys=["idx", input_key, output_key],
                    new_keys=["idx", "input", "output"],
                )
            )

        ### INPUT WRAPPERS ###
        if input_wrappers is not None and len(input_wrappers) > 0:
            self._input_wrappers = input_wrappers
        else:
            self._input_wrappers = [DummyInputWrapper()]

        for i in reversed(range(len(self._input_wrappers))):
            # self._input_wrappers[i].set_call_run_parameters_history_update_fn(self._update_run_parameters_history)
            if i == len(self._input_wrappers) - 1:
                self._input_wrappers[i].initialize(
                    output_space=self._system.input_space
                )
                output_key = f"run_parameters"
            else:
                self._input_wrappers[i].initialize(
                    output_space=self._input_wrappers[i + 1].input_space
                )
                output_key = f"run_parameters_{i}"

            if i == 0:
                input_key = "raw_run_parameters"
            else:
                input_key = f"run_parameters_{i-1}"

            self._input_wrappers[i].set_history_access_fn(
                access_history_fn(
                    keys=["idx", input_key, output_key],
                    new_keys=["idx", "input", "output"],
                )
            )

        ### EXPLORER ###
        self._explorer = explorer
        self._explorer.set_call_output_history_update_fn(self._update_outputs_history)
        # self._explorer.set_call_run_parameters_history_update_fn(self._update_run_parameters_history)
        self._explorer.initialize(
            input_space=self._output_representations[-1].output_space,
            output_space=self._input_wrappers[0].input_space,
            input_distance_fn=self._output_representations[-1].calc_distance,
        )
        self._explorer.set_history_access_fn(
            access_history_fn(
                keys=["idx", "output", "raw_run_parameters"],
                new_keys=["idx", "input", "output"],
            )
        )

        self._action_policy = action_policy
        self._on_discovery_callbacks = on_discovery_callbacks
        self._on_save_finished_callbacks = on_save_finished_callbacks
        self._on_finished_callbacks = on_finished_callbacks
        self._on_cancelled_callbacks = on_cancelled_callbacks
        self._on_error_callbacks = on_error_callbacks
        self._on_save_callbacks = on_save_callbacks
        self.interact_callbacks = interact_callbacks
        self.cancellation_token = CancellationToken()

    def _process_output(
        self,
        output: typing.Dict[str, torch.Tensor],
        document_id: int,
        starting_index: int = 0,
        is_output_new_discovery: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Process the output and store it in the tinyDB to make it usable in different modules of the experiment

        Args:
            output: current output
            document_id: current document id
            starting_index: first index to consider in the outputs representations list
            is_output_new_discovery: indiacte if the current output come from a new discovery

        Returns:
            output: The current output after it was processed
        """
        for i, output_representation in enumerate(
            self._output_representations[starting_index:]
        ):
            output = output_representation.map(copy(output), is_output_new_discovery)
            if i == len(self._output_representations) - 1:
                self.db.update({"output": copy(output)}, doc_ids=[document_id])
            else:
                self.db.update({f"output_{i}": copy(output)}, doc_ids=[document_id])
        return output

    def _update_outputs_history(self, output_representation_idx):
        """
        Iterate over history and update values of outputs produced after `output_representation_idx`.

        Args:
            output_representation_idx: index from which we will start updating
        """
        for document in self.db:
            if output_representation_idx == 0:
                # starting from first output => raw_output
                output = document["raw_output"]
            else:
                output = document[f"output_{output_representation_idx-1}"]
            self._process_output(
                output,
                document.doc_id,
                starting_index=output_representation_idx,
                is_output_new_discovery=False,
            )

    def _process_run_parameters(
        self,
        run_parameters: Dict[str, Any],
        document_id: int,
        starting_index: int = 0,
        is_input_new_discovery: bool = True,
    ) -> Dict[str, Any]:
        """
        Process the run_parameters and store it in the tinyDB to make it usable in different modules of the experiment

        Args:
            run_parameters: current run_parameters
            document_id: current document id
            starting_index: first index to consider in the outputs representations list
            is_input_new_discovery: indiacte if the current input come from a new discovery

        Returns:
            output: The current output after it was processed
        """

        for i, input_wrapper in enumerate(self._input_wrappers[starting_index:]):
            run_parameters = input_wrapper.map(
                copy(run_parameters), is_input_new_discovery
            )
            if i == len(self._input_wrappers) - 1:
                self.db.update(
                    {"run_parameters": copy(run_parameters)}, doc_ids=[document_id]
                )
            else:
                self.db.update(
                    {f"run_parameters_{i}": copy(run_parameters)}, doc_ids=[document_id]
                )
        return run_parameters

    # def _update_run_parameters_history(self, run_parameters_idx):
    #     '''
    #         Iterate over history and update values of run_parameters produced after `run_parameters_idx`.
    #     '''
    #     for document in self.db.all():
    #         if run_parameters_idx == 0:
    #             run_parameters =  document['raw_run_parameters'] # starting from first run_parameters => raw_run_parameters
    #         else:
    #             run_parameters = document[f'run_parameters_{run_parameters_idx-1}']
    #         self._process_run_parameters(run_parameters, document.doc_id, starting_index=run_parameters_idx, is_input_new_discovery=False)

    def _raise_callbacks(self, callbacks: typing.List[Callable], **kwargs) -> None:
        """
        Raise all callbacks linked to the current event (new discovery, a save, the end of the experiment...)

        Args:
            callbacks: list of all callbacks must be raise
            kwargs: some usefull parameters like run_idx, seed, experiment_id, some modules...
        """
        for callback in callbacks:
            callback(pipeline=self, **kwargs)

    def run(self, n_exploration_runs: int) -> None:
        """
        Launches the experiment for `n_exploration_runs` explorations.

        Args:
            n_exploration_runs: number of explorations
        """
        run_idx = 0
        BaseAutoDiscModule.CURRENT_RUN_INDEX = 0
        system_steps = [0]
        Interact.init_seed(
            self.interact_callbacks,
            {"experiment_id": self.experiment_id, "seed": self.seed, "idx": 0},
        )
        try:
            while run_idx < n_exploration_runs:
                if self.cancellation_token.get():
                    break

                raw_run_parameters = self._explorer.sample()
                document_id = self.db.insert(
                    {"idx": run_idx, "raw_run_parameters": copy(raw_run_parameters)}
                )
                with torch.no_grad():
                    run_parameters = self._process_run_parameters(
                        raw_run_parameters, document_id
                    )

                o, r, d, i = self._system.reset(copy(run_parameters)), 0, None, False

                while not d:
                    if self._action_policy is not None:
                        a = self._action_policy(o, r)
                    else:
                        a = None

                    with torch.no_grad():
                        o, r, d, i = self._system.step(a)
                    system_steps[-1] += 1

                with torch.no_grad():
                    raw_output = self._system.observe()
                    self.db.update(
                        {"raw_output": copy(raw_output)}, doc_ids=[document_id]
                    )
                    output = self._process_output(raw_output, document_id)
                    rendered_output = self._system.render()

                self._explorer.observe(copy(raw_run_parameters), copy(output))

                self._raise_callbacks(
                    self._on_discovery_callbacks,
                    run_idx=run_idx,
                    seed=self.seed,
                    raw_run_parameters=raw_run_parameters,
                    run_parameters=run_parameters,
                    raw_output=raw_output,
                    output=output,
                    rendered_output=rendered_output,
                    experiment_id=self.experiment_id,
                )
                self._system.logger.info(
                    "[DISCOVERY] - New discovery from experiment {} with seed {}".format(
                        self.experiment_id, self.seed
                    )
                )
                self._explorer.optimize()  # TODO callbacks

                if (
                    run_idx + 1
                ) % self.save_frequency == 0 or run_idx + 1 == n_exploration_runs:
                    self._raise_callbacks(
                        self._on_save_callbacks,
                        run_idx=run_idx,
                        seed=self.seed,
                        experiment_id=self.experiment_id,
                        system=self._system,
                        explorer=self._explorer,
                        input_wrappers=self._input_wrappers,
                        output_representations=self._output_representations,
                        in_memory_db=self.db,
                    )
                    self._raise_callbacks(
                        self._on_save_finished_callbacks,
                        run_idx=run_idx,
                        seed=self.seed,
                        experiment_id=self.experiment_id,
                    )
                    self._system.logger.info(
                        "[SAVED] - experiment {} with seed {} saved".format(
                            self.experiment_id, self.seed
                        )
                    )

                run_idx += 1
                BaseAutoDiscModule.CURRENT_RUN_INDEX += 1

        except Exception as ex:
            message = "error in experiment {} run_idx {} seed {} = {}".format(
                self.experiment_id, run_idx, self.seed, traceback.format_exc()
            )
            if len(message) > 8000:  # Cut message to match varchar length of AppDB
                message = message[:7997] + "..."
            self._system.logger.error("[ERROR] - " + message)
            self._raise_callbacks(
                self._on_error_callbacks,
                run_idx=run_idx,
                seed=self.seed,
                experiment_id=self.experiment_id,
            )
            self.db.close()
            raise

        self._system.close()
        if self.cancellation_token.get():
            self._system.logger.info(
                "[CANCELLED] - experiment {} with seed {} cancelled".format(
                    self.experiment_id, self.seed
                )
            )
        else:
            self._system.logger.info(
                "[FINISHED] - experiment {} with seed {} finished".format(
                    self.experiment_id, self.seed
                )
            )
        self._raise_callbacks(
            self._on_cancelled_callbacks
            if self.cancellation_token.get()
            else self._on_finished_callbacks,
            run_idx=run_idx,
            seed=self.seed,
            experiment_id=self.experiment_id,
        )
        self.db.close()
