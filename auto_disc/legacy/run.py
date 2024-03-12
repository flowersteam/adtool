import argparse
import json
import os
import random
import sys
from typing import Callable, Dict, List

import numpy as np
import torch
from auto_disc.legacy import ExperimentPipeline
from auto_disc.legacy.registration import REGISTRATION
from auto_disc.legacy.utils.callbacks import interact_callbacks
from auto_disc.legacy.utils.logger import AutoDiscLogger

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, "../"))


def create(
    parameters: Dict,
    experiment_id: int,
    seed: int,
    additional_callbacks: Dict[str, List[Callable]] = None,
    additional_handlers: List[AutoDiscLogger] = None,
    interactMethod: Callable = None,
) -> ExperimentPipeline:
    """
    Setup the whole experiment. Set each modules, logger, callbacks and use them to define the experiment pipeline.

    Args:
        parameters: Experiment config (define wich systems wich explorer wich callbacks and all other information needed to set an experiment)
        experiment_id: Current experiment id
        seed: current seed number
        additional_callbacks: callbacks we want use in addition to callbacks from parameters arguments
        additional_handlers: handlers we want to use in addition to logger_handlers from parameters arguments

    Returns:
        experiment: The experiment we have just defined
    """

    _set_seed(seed)
    save_frequency = parameters["experiment"]["save_frequency"]

    # Get logger
    handlers = []
    for logger_handler in parameters["logger_handlers"]:
        hanlder_key = logger_handler["name"]
        handler_class = REGISTRATION["logger_handlers"][hanlder_key]
        handler = handler_class(**logger_handler["config"], experiment_id=experiment_id)
        handlers.append(handler)
    if additional_handlers is not None:
        handlers.extend(additional_handlers)

    logger = AutoDiscLogger(experiment_id, seed, handlers)

    # Get explorer
    explorer_class = REGISTRATION["explorers"][parameters["explorer"]["name"]]
    explorer = explorer_class(logger=logger, **parameters["explorer"]["config"])

    # Get system
    system_class = REGISTRATION["systems"][parameters["system"]["name"]]
    system = system_class(logger=logger, **parameters["system"]["config"])

    # Get input wrappers
    input_wrappers = []
    for _input_wrapper in parameters["input_wrappers"]:
        input_wrapper_class = REGISTRATION["input_wrappers"][_input_wrapper["name"]]
        input_wrappers.append(
            input_wrapper_class(logger=logger, **_input_wrapper["config"])
        )

    # Get output representations
    output_representations = []
    for _output_representation in parameters["output_representations"]:
        output_representation_class = REGISTRATION["output_representations"][
            _output_representation["name"]
        ]
        output_representations.append(
            output_representation_class(
                logger=logger, **_output_representation["config"]
            )
        )

    # Get callbacks
    callbacks = {
        "on_discovery": [],
        "on_save_finished": [],
        "on_finished": [],
        "on_error": [],
        "on_cancelled": [],
        "on_saved": [],
        "interact": {},
    }

    for callback_key in callbacks:
        if additional_callbacks is not None:
            if callback_key != "interact":
                callbacks[callback_key].extend(additional_callbacks[callback_key])
            else:
                callbacks[callback_key].update(additional_callbacks[callback_key])
        for _callback in parameters["callbacks"][callback_key]:
            callback_class = REGISTRATION["callbacks"][callback_key][_callback["name"]]
            if callback_key == "interact":
                callbacks[callback_key].update(
                    {
                        _callback["name"]: callback_class(
                            logger=logger,
                            interactMethod=interactMethod,
                            **_callback["config"]
                        )
                    }
                )
            else:
                callbacks[callback_key].append(
                    callback_class(logger=logger, **_callback["config"])
                )

    # Create experiment pipeline
    experiment = ExperimentPipeline(
        experiment_id=experiment_id,
        seed=seed,
        save_frequency=save_frequency,
        system=system,
        explorer=explorer,
        input_wrappers=input_wrappers,
        output_representations=output_representations,
        on_discovery_callbacks=callbacks["on_discovery"],
        on_save_finished_callbacks=callbacks["on_save_finished"],
        on_finished_callbacks=callbacks["on_finished"],
        on_cancelled_callbacks=callbacks["on_cancelled"],
        on_save_callbacks=callbacks["on_saved"],
        on_error_callbacks=callbacks["on_error"],
        interact_callbacks=callbacks["interact"],
    )

    return experiment


def start(experiment: ExperimentPipeline, nb_iterations: int) -> None:
    """
    Runs an experiment for a number of iterations

    Args:
        experiment: The experiment we want to launch
        nb_iterations: the number explorations
    """
    experiment.run(nb_iterations)


def _set_seed(seed: int) -> None:
    """
    Set torch seed to make experiment repeatable.

    Args:
        seed: seed number
    """
    seed = int(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--experiment_id", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--nb_iterations", type=int, required=True)

    args = parser.parse_args()

    with open(args.config_file) as json_file:
        config = json.load(json_file)

    experiment = create(config, args.experiment_id, args.seed)
    start(experiment, args.nb_iterations)
