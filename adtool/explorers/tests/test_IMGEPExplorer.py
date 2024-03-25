import os
import pathlib
import shutil

import pytest
import torch
from adtool.maps.MeanBehaviorMap import MeanBehaviorMap
from adtool.maps.UniformParameterMap import UniformParameterMap
from adtool.explorers.IMGEPExplorer import IMGEPExplorer, IMGEPFactory


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


def add_gaussian_noise_test(
    input_tensor: torch.Tensor,
    mean: torch.Tensor = torch.tensor([10000.0]),
    std: torch.Tensor = torch.tensor([1.0]),
) -> torch.Tensor:
    if not isinstance(mean, torch.Tensor):
        mean = torch.tensor(mean, dtype=float)
    if not isinstance(std, torch.Tensor):
        std = torch.tensor(std, dtype=float)
    noise_unit = torch.randn(input_tensor.size())
    noise = noise_unit * std + mean
    return input_tensor + noise


def test_IMGEPExplorer___init__():
    mean_map = MeanBehaviorMap(premap_key="output")
    param_map = UniformParameterMap(
        premap_key="params",
        tensor_low=torch.tensor([0.0, 0.0, 0.0]),
        tensor_high=torch.tensor([2.0, 2.0, 2.0]),
    )
    explorer = IMGEPExplorer(
        premap_key="output",
        postmap_key="params",
        parameter_map=param_map,
        behavior_map=mean_map,
        equil_time=2,
    )
    assert explorer.behavior_map == mean_map
    assert explorer.parameter_map == param_map
    assert explorer.equil_time == 2
    assert explorer.timestep == 0


def test_IMGEPExplorer__extract_tensor_history():
    history_buffer = [
        {
            "params": torch.tensor([0.9623, 0.8531, 0.2911]),
            "equil": 1,
            "output": torch.tensor([0.6001]),
        },
        {
            "params": torch.tensor([0.9673, 0.7330, 0.8939]),
            "equil": 1,
            "output": torch.tensor([0.6000]),
        },
        {
            "params": torch.tensor([0.0185, 0.4275, 0.6428]),
            "equil": 1,
            "output": torch.tensor([0.8872]),
        },
    ]
    mean_map = MeanBehaviorMap(premap_key="output")
    param_map = UniformParameterMap(
        premap_key="params",
        tensor_low=torch.tensor([0.0, 0.0, 0.0]),
        tensor_high=torch.tensor([2.0, 2.0, 2.0]),
    )
    explorer = IMGEPExplorer(
        premap_key="output",
        postmap_key="params",
        parameter_map=param_map,
        behavior_map=mean_map,
        equil_time=2,
    )
    param_tensor = explorer._extract_tensor_history(history_buffer, "params")
    param_history = torch.tensor(
        [[0.9623, 0.8531, 0.2911], [0.9673, 0.7330, 0.8939], [0.0185, 0.4275, 0.6428]]
    )
    assert torch.allclose(param_tensor, param_history)
    output_tensor = explorer._extract_tensor_history(history_buffer, "output")
    output_history = torch.tensor([[0.6001], [0.6000], [0.8872]])
    assert torch.allclose(output_tensor, output_history)


def test_IMGEPExplorer__find_closest():
    goal_history = torch.tensor([[1, 2, 3], [4, 5, 6], [-1, -1, -1]], dtype=float)
    goal = torch.tensor([-0.99, -0.99, -0.99])

    mean_map = MeanBehaviorMap(premap_key="output")
    param_map = UniformParameterMap(
        premap_key="params",
        tensor_low=torch.tensor([0.0, 0.0, 0.0]),
        tensor_high=torch.tensor([2.0, 2.0, 2.0]),
    )
    explorer = IMGEPExplorer(
        premap_key="output",
        postmap_key="params",
        parameter_map=param_map,
        behavior_map=mean_map,
        equil_time=2,
    )

    idx = explorer._find_closest(goal, goal_history)
    assert idx == torch.tensor([2])
    assert torch.allclose(goal_history[idx], torch.tensor([-1.0, -1.0, -1.0]))


def test_IMGEPExplorer_observe_results():
    mean_map = MeanBehaviorMap(premap_key="output")
    param_map = UniformParameterMap(
        premap_key="params",
        tensor_low=torch.tensor([0.0, 0.0, 0.0]),
        tensor_high=torch.tensor([2.0, 2.0, 2.0]),
    )
    explorer = IMGEPExplorer(
        premap_key="output",
        postmap_key="params",
        parameter_map=param_map,
        behavior_map=mean_map,
        equil_time=2,
    )

    output_tensor = torch.tensor([1.0, 1.0, 1.0])
    system_output = {"metadata": 1, "output": output_tensor}
    system_output = explorer.observe_results(system_output)

    assert torch.allclose(output_tensor, system_output["output"])
    # check mutability
    assert system_output["output"] is not output_tensor
    output_tensor += 1
    assert not torch.allclose(output_tensor, system_output["output"])


def test_IMGEPExplorer_bootstrap():
    mean_map = MeanBehaviorMap(premap_key="output")
    param_map = UniformParameterMap(
        premap_key="params",
        tensor_low=torch.tensor([0.0, 0.0, 0.0]),
        tensor_high=torch.tensor([6.0, 6.0, 6.0]),
    )
    explorer = IMGEPExplorer(
        premap_key="output",
        postmap_key="params",
        parameter_map=param_map,
        behavior_map=mean_map,
        equil_time=2,
    )
    init_dict = explorer.bootstrap()
    assert init_dict.get("params", None) is not None
    assert init_dict["equil"] == 1


def test_IMGEPExplorer_map():
    # TODO: mock the behavior and parameter maps
    mean_map = MeanBehaviorMap(premap_key="output")
    param_map = UniformParameterMap(
        premap_key="params",
        tensor_low=torch.tensor([0.0, 0.0, 0.0]),
        tensor_high=torch.tensor([6.0, 6.0, 6.0]),
    )
    explorer = IMGEPExplorer(
        premap_key="output",
        postmap_key="params",
        parameter_map=param_map,
        behavior_map=mean_map,
        equil_time=2,
    )
    system_output = {
        "metadata": 1,
        "params": torch.tensor([2.0, 2.0, 2.0]),
        "output": torch.tensor([1.0, 2.0, 3.0]),
    }

    new_params = explorer.map(system_output)
    assert new_params["params"].size() == torch.Size([3])
    assert torch.allclose(new_params["raw_output"], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.allclose(new_params["output"], torch.tensor([2.0]))
    assert explorer.timestep == 1

    # check mutability
    assert new_params is not system_output
    system_output["metadata"] = 2
    assert new_params["metadata"] == 1


def test_IMGEPExplorer_read_last_discovery():
    pass


def test_IMGEPExplorer_suggest_trial_behavioral_diffusion():
    mean_map = MeanBehaviorMap(premap_key="output")
    param_map = UniformParameterMap(
        premap_key="params",
        tensor_low=torch.tensor([0.0, 0.0, 0.0]),
        tensor_high=torch.tensor([6.0, 6.0, 6.0]),
    )
    explorer = IMGEPExplorer(
        premap_key="output",
        postmap_key="params",
        parameter_map=param_map,
        behavior_map=mean_map,
        equil_time=2,
    )

    # mock history
    mock_discovery_history = [
        {
            "metadata": 1,
            "params": torch.tensor([0.0, 1.0, 2.0]),
            "raw_output": torch.tensor([1.0, 2.0, 3.0]),
            "output": torch.tensor([2.0]),
        },
        {
            "metadata": 1,
            "params": torch.tensor([3.0, 4.0, 5.0]),
            "raw_output": torch.tensor([4.0, 5.0, 6.0]),
            "output": torch.tensor([5.0]),
        },
        {
            "metadata": 1,
            "params": torch.tensor([-1.0, -1.0, -1]),
            "raw_output": torch.tensor([0.0, 0.0, 0.0]),
            "output": torch.tensor([0.0]),
        },
    ]
    explorer._history_saver.buffer = mock_discovery_history
    explorer.behavior_map.projector.low = torch.tensor([0.0])
    explorer.behavior_map.projector.high = torch.tensor([5.0])
    explorer.behavior_map.projector.tensor_shape = torch.Size([1])
    explorer.timestep = 2
    explorer.mutator = add_gaussian_noise_test

    # actual test
    # NOTE: not deterministic because of noise
    # in real use, the resource_uri would be set when
    # the explorer is bound as a submodule of a parent module
    explorer.locator.resource_uri = RESOURCE_URI
    explorer._history_saver.locator.resource_uri = RESOURCE_URI

    params_trial = explorer.suggest_trial()
    assert params_trial.size() == torch.Size([3])
    assert torch.mean(params_trial) > 100

    # suggest_trial does not increment timestep
    assert explorer.timestep == 2


def test_IMGEPExplorer_suggest_trial_behavioral_diffusion_arbitrary_buf():
    """
    This test is the same as the above, but we do a save step
    to make sure that the test will pass when retrieving buffers
    from storage.
    """

    mean_map = MeanBehaviorMap(premap_key="output")
    param_map = UniformParameterMap(
        premap_key="params",
        tensor_low=torch.tensor([0.0, 0.0, 0.0]),
        tensor_high=torch.tensor([6.0, 6.0, 6.0]),
    )
    explorer = IMGEPExplorer(
        premap_key="output",
        postmap_key="params",
        parameter_map=param_map,
        behavior_map=mean_map,
        equil_time=2,
    )

    # mock history
    mock_discovery_history = [
        {
            "metadata": 1,
            "params": torch.tensor([0.0, 1.0, 2.0]),
            "raw_output": torch.tensor([1.0, 2.0, 3.0]),
            "output": torch.tensor([2.0]),
        },
        {
            "metadata": 1,
            "params": torch.tensor([3.0, 4.0, 5.0]),
            "raw_output": torch.tensor([4.0, 5.0, 6.0]),
            "output": torch.tensor([5.0]),
        },
        {
            "metadata": 1,
            "params": torch.tensor([-1.0, -1.0, -1]),
            "raw_output": torch.tensor([0.0, 0.0, 0.0]),
            "output": torch.tensor([0.0]),
        },
    ]
    explorer._history_saver.buffer = mock_discovery_history
    explorer.behavior_map.projector.low = torch.tensor([0.0])
    explorer.behavior_map.projector.high = torch.tensor([5.0])
    explorer.behavior_map.projector.tensor_shape = torch.Size([1])
    explorer.timestep = 2
    explorer.mutator = add_gaussian_noise_test

    # save to storage
    # NOTE: as usual, calling save_leaf binds the module to a resource_uri
    # (importantly, the explorer._history_saver.locator.resource_uri is set to
    # the same when explorer._history_saver.save_leaf is recursively called)
    explorer.save_leaf(RESOURCE_URI)
    assert len(explorer._history_saver.buffer) == 0
    assert len(os.listdir(RESOURCE_URI)) != 0

    # actual test
    # NOTE: not deterministic because of noise

    params_trial = explorer.suggest_trial()
    assert params_trial.size() == torch.Size([3])
    assert torch.mean(params_trial) > 100

    # suggest_trial does not increment timestep
    assert explorer.timestep == 2


def test_IMGEPFactory___init__():
    factory = IMGEPFactory()
    # test config params set non None
    assert factory.config


def test_IMGEPFactory___call__():
    # tests that the factory settings are passed to the explorer
    factory = IMGEPFactory(equil_time=5)
    explorer = factory()
    assert explorer.equil_time == 5


def test_IMGEPFactory_make_parameter_map():
    # test default initialization
    factory = IMGEPFactory()
    param_map = factory.make_parameter_map()
    assert isinstance(param_map, UniformParameterMap)

    # test exception if parameter map not found
    with pytest.raises(Exception):
        factory = IMGEPFactory(parameter_map="test")


def test_IMGEPFactory_make_behavior_map():
    # test default initialization
    factory = IMGEPFactory()
    behavior_map = factory.make_behavior_map()
    assert isinstance(behavior_map, MeanBehaviorMap)

    # test custom initialization
    factory = IMGEPFactory(behavior_map_config={"premap_key": "test"})
    behavior_map = factory.make_behavior_map()
    assert isinstance(behavior_map, MeanBehaviorMap)
    assert behavior_map.premap_key == "test"

    # test exception if behavior map not found
    with pytest.raises(Exception):
        factory = IMGEPFactory(behavior_map="test")


def test_IMGEPFactory_make_mutator():
    factory = IMGEPFactory()
    mutator = factory.make_mutator()
    assert callable(mutator)
