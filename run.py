"""
Helper script which allows creation of `ExperimentPipeline`.
This file `run.py` can also be run as `__main__`,
for example in remote configurations.
"""
import argparse
import json
import random
from typing import Callable, Dict, List

import numpy as np
import torch
from adtool.ExperimentPipeline import ExperimentPipeline
from adtool.auto_disc.utils.logger import AutoDiscLogger
from adtool.utils.leafutils.leafstructs.registration import get_cls_from_name
from mergedeep import merge


def create(
    parameters: Dict,
    experiment_id: int,
    seed: int,
    additional_callbacks: Dict[str, List[Callable]] = None,
    additional_handlers: List[AutoDiscLogger] = None,
    interactMethod: Callable = None,
) -> ExperimentPipeline:
    """
    Setup the whole experiment. Set each modules, logger, callbacks and use
    them to define the experiment pipeline.

    In addition to callbacks and handlers defined in `parameters`,
    you can pass extras. as keyword arguments.

    #### Args:
    - parameters: Experiment config (define which systems which explorer
    which callbacks and all other information needed to set an experiment)
    - experiment_id: Current experiment id
    - seed: current seed number
    - additional_callbacks
    - additional_handlers

    #### Returns:
    - experiment: The experiment we have just defined
    """

    _set_seed(seed)
    # Get logger
    # FIXME: broken with get_cls_from_name, need to
    # add handlers to registration
    handlers = []
    for logger_handler in parameters["logger_handlers"]:
        handler_class = get_cls_from_name(logger_handler["name"], "handlers")
        handler = handler_class(**logger_handler["config"], experiment_id=experiment_id)
        handlers.append(handler)
    if additional_handlers is not None:
        handlers.extend(additional_handlers)

    logger = AutoDiscLogger(experiment_id, seed, handlers)

    # Get callbacks
    callbacks = {
        "on_discovery": [],
        "on_save_finished": [],
        "on_finished": [],
        "on_error": [],
        "on_cancelled": [],
        "on_saved": [],
    }
    # initialize callbacks which require lookup
    # NOTE: stateful callbacks are deprecated, and new callbacks simply have a
    # dummy __init__ to obey this interface

    # FIXME: null guard
    if len(parameters["callbacks"]) > 0:
        cb_request = parameters["callbacks"]
        for cb_key in cb_request.keys():
            for cb in cb_request[cb_key]:
                cb_config = cb["config"]
                type_name = "callbacks." + cb_key
                callback = get_cls_from_name(cb["name"], ad_type_name=type_name)
                # initialize callback instance
                callbacks[cb_key].append(callback(**cb_config))

    # add additional callbacks which are already initialized Callables
    if additional_callbacks:
        for cb_key, lst in additional_callbacks.items():
            callbacks[cb_key] += lst

    # short circuit if "resume_from_uid" is set
    resume_ckpt = parameters["experiment"]["config"].get("resume_from_uid", None)
    if resume_ckpt is not None:
        resource_uri = parameters["experiment"]["config"]["save_location"]
        experiment = ExperimentPipeline().load_leaf(
            uid=resume_ckpt, resource_uri=resource_uri
        )

        # set attributes pruned by save_leaf
        experiment.logger = logger
        experiment._on_discovery_callbacks = callbacks["on_discovery"]
        experiment._on_save_finished_callbacks = callbacks["on_save_finished"]
        experiment._on_finished_callbacks = callbacks["on_finished"]
        experiment._on_cancelled_callbacks = callbacks["on_cancelled"]
        experiment._on_save_callbacks = callbacks["on_saved"]
        experiment._on_error_callbacks = callbacks["on_error"]
        # experiment._interact_callbacks = callbacks['interact']

        return experiment

    # Get explorer factory and generate explorer
    explorer_factory_class = get_cls_from_name(
        parameters["explorer"]["name"], "explorers"
    )
    explorer_factory = explorer_factory_class(**parameters["explorer"]["config"])
    explorer = explorer_factory()

    # Get system
    system_class = get_cls_from_name(parameters["system"]["name"], "systems")
    system = system_class(**parameters["system"]["config"])

    # Create experiment pipeline
    experiment = ExperimentPipeline(
        experiment_id=experiment_id,
        seed=seed,
        save_frequency=parameters["experiment"]["config"]["save_frequency"],
        system=system,
        explorer=explorer,
        on_discovery_callbacks=callbacks["on_discovery"],
        on_save_finished_callbacks=callbacks["on_save_finished"],
        on_finished_callbacks=callbacks["on_finished"],
        on_cancelled_callbacks=callbacks["on_cancelled"],
        on_save_callbacks=callbacks["on_saved"],
        on_error_callbacks=callbacks["on_error"],
        logger=logger,
        resource_uri=parameters["experiment"]["config"]["save_location"],
    )

    return experiment


def start(experiment: ExperimentPipeline, nb_iterations: int) -> None:
    """
    Runs an experiment for a number of iterations

    #### Args:
    - experiment: The experiment we want to launch
    - nb_iterations: the number explorations
    """
    experiment.run(nb_iterations)


def _set_seed(seed: int) -> None:
    """
    Set torch seed to make experiment repeatable.

    #### Args:
    - seed: seed number
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
    parser.add_argument("--experiment_id", type=int, required=False, default= 0)
    parser.add_argument("--seed", type=int, required=False, default=42)
    parser.add_argument("--nb_iterations", type=int, required=False, default=30)
                        
    args = parser.parse_args()

    with open(args.config_file) as json_file:
        config = json.load(json_file)

    experiment = create(config, args.experiment_id, args.seed)
    start(experiment, args.nb_iterations)
