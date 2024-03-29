import os
import pathlib
import shutil

import torch
from adtool.maps.MeanBehaviorMap import MeanBehaviorMap
from adtool.maps.UniformParameterMap import UniformParameterMap
from examples.exponential_mixture.systems.ExponentialMixture import ExponentialMixture
from adtool.explorers.IMGEPExplorer import IMGEPExplorer, IMGEPFactory
from adtool.wrappers.IdentityWrapper import IdentityWrapper
from adtool.ExperimentPipeline import ExperimentPipeline
from adtool.callbacks.on_save_callbacks.save_leaf_callback import (
    SaveLeaf,
)
from adtool.utils.logger import AutoDiscLogger


def setup_function(function):
    global RESOURCE_URI
    file_path = str(pathlib.Path(__file__).parent.resolve())
    RESOURCE_URI = os.path.join(file_path, "tmp")
    os.mkdir(RESOURCE_URI)
    return


def teardown_function(function):
    global RESOURCE_URI
    if os.path.exists(RESOURCE_URI):
        shutil.rmtree(RESOURCE_URI)
    return


def callback(**__kwargs):
    print(f"Callback was called on pipeline with" f"kwargs {__kwargs}")
    return


def test_dummy_callback(capsys):
    callback(config=1, metadata=2)
    captured = capsys.readouterr()
    assert captured.out != ""


def test___init__():
    experiment_id = 1
    seed = 1
    system_input_key = "params"
    system_output_key = "output"

    system = ExponentialMixture(sequence_max=10, sequence_density=10)
    mean_map = MeanBehaviorMap(premap_key=system_output_key)
    param_map = UniformParameterMap(
        premap_key=system_input_key,
        tensor_low=torch.tensor([0.0, 0.0, 0.0]),
        tensor_high=torch.tensor([3.0, 3.0, 3.0]),
    )
    explorer = IMGEPExplorer(
        premap_key=system_output_key,
        postmap_key=system_input_key,
        parameter_map=param_map,
        behavior_map=mean_map,
        equil_time=2,
    )

    pipeline = ExperimentPipeline(
        experiment_id=experiment_id, seed=seed, system=system, explorer=explorer
    )
    uid = pipeline.save_leaf(resource_uri=RESOURCE_URI)
    x = ExperimentPipeline()
    new_pipeline = x.load_leaf(uid, resource_uri=RESOURCE_URI)

    # check explorer state
    assert new_pipeline._explorer.premap_key == system_output_key
    assert new_pipeline._explorer.postmap_key == system_input_key
    assert torch.allclose(
        new_pipeline._explorer.parameter_map.projector.low,
        torch.tensor([0.0, 0.0, 0.0]),
    )
    assert torch.allclose(
        new_pipeline._explorer.parameter_map.projector.high,
        torch.tensor([3.0, 3.0, 3.0]),
    )
    # check system type
    assert isinstance(new_pipeline._system, ExponentialMixture)
    # check container pointers
    assert new_pipeline._explorer._container_ptr == new_pipeline

    # TODO: finish the checks of state, but i'm lazy


def test_run():
    experiment_id = 1
    seed = 1
    system_input_key = "params"
    system_output_key = "output"
    logger = AutoDiscLogger(experiment_id, seed, [])

    factory = IMGEPFactory(
        equil_time=5,
        parameter_map_config={"tensor_low": 0.0, "tensor_high": 1.0},
        mutator="gaussian",
        mutator_config={"std": 0.0},
    )
    explorer = factory()
    system = ExponentialMixture(sequence_density=10)

    uid_capture = []

    # callbacks
    def retrieve_uid_callback():
        def callback(**kwargs):
            uid = kwargs["uid"]
            uid_capture.append(uid)
            return

        return callback

    pipeline = ExperimentPipeline(
        experiment_id=experiment_id,
        seed=seed,
        system=system,
        explorer=explorer,
        on_discovery_callbacks=[callback],
        on_save_callbacks=[SaveLeaf()],
        on_save_finished_callbacks=[retrieve_uid_callback()],
        logger=logger,
        save_frequency=100,
        resource_uri=RESOURCE_URI,
    )

    pipeline.run(20)

    assert len(uid_capture) == 1  # only one save within 20 steps

    new_pipeline = ExperimentPipeline().load_leaf(
        uid=uid_capture[0], resource_uri=RESOURCE_URI
    )
    history_buffer = new_pipeline._explorer._history_saver.buffer

    # check equilibriation phase
    seen_params = set()
    for i in range(5):
        assert history_buffer[i]["equil"] == 1
        seen_params.add(history_buffer[i][system_input_key])

    # check exploration phase
    for i in range(5, 20):
        assert history_buffer[i]["equil"] == 0

        # for IMGEP without mutator, it is stuck in an orbit
        match_count = 0
        for p in seen_params:
            match_count += int(torch.allclose(history_buffer[i][system_input_key], p))
        assert match_count == 1


# TODO: fix loggers, see gh #45
# def test_logger(capsys):
#     experiment_id = 1
#     seed = 1
#     logger = AutoDiscLogger(experiment_id, seed, [])
#     system = ExponentialMixture(sequence_density=10)
#     explorer_factory = IMGEPFactory(
#         equil_time=5, param_dim=3, param_init_low=0.0, param_init_high=1.0
#     )
#     explorer = explorer_factory()
#     input_pipeline = IdentityWrapper()
#     output_pipeline = IdentityWrapper()
#     pipeline = ExperimentPipeline(
#         experiment_id=experiment_id,
#         seed=seed,
#         system=system,
#         explorer=explorer,
#         input_pipeline=input_pipeline,
#         output_pipeline=output_pipeline,
#         on_save_callbacks=[callback],
#         logger=logger,
#     )
#     pipeline.logger.info("testing")
#     expected_str = "ad_tool_logger - INFO - SEED 1 - LOG_ID 1_1_1 - testing\n"
#     captured = capsys.readouterr()
#     assert captured.err == expected_str
