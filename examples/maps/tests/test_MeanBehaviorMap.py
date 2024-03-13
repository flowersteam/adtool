import os
import pathlib
import shutil

import torch
from adtool.auto_disc.maps.MeanBehaviorMap import MeanBehaviorMap
from adtool.auto_disc.maps.UniformParameterMap import UniformParameterMap


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


def test_map():
    # test for a flat 1D tensor
    input_dict = {"metadata": 1, "output": torch.rand(10)}
    mean_map = MeanBehaviorMap(premap_key="output")
    output_dict = mean_map.map(input_dict)
    assert output_dict["output"].size() == torch.Size([1])
    assert mean_map.projector.low is not None
    assert mean_map.projector.high is not None

    # test for a 2D tensor
    input_dict = {"metadata": 1, "output": torch.rand(10, 10)}
    mean_map = MeanBehaviorMap(premap_key="output")
    output_dict = mean_map.map(input_dict)
    assert output_dict["output"].size() == torch.Size([1])
    assert mean_map.projector.low is not None
    assert mean_map.projector.high is not None


def test_sample():
    # redundant test, see test for BoxProjector.map and BoxProjector.sample
    pass


def test_save():
    input_dict = {"metadata": 1, "output": torch.tensor([2.0, 2.0, 2.0, 2.0, 2.0])}
    mean_map = MeanBehaviorMap(premap_key="output")
    output_dict = mean_map.map(input_dict)
    uid = mean_map.save_leaf(resource_uri=RESOURCE_URI)

    new_map = MeanBehaviorMap()
    loaded_map = new_map.load_leaf(uid, resource_uri=RESOURCE_URI)
    assert torch.allclose(
        loaded_map.projector.low, torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])
    )
    assert torch.allclose(
        loaded_map.projector.high, torch.tensor([2.0, 2.0, 2.0, 2.0, 2.0])
    )
