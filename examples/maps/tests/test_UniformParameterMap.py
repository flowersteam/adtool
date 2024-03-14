import os
import pathlib
import shutil

import torch
from adtool.maps.UniformParameterMap import UniformParameterMap


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


def test___init__():
    # default init
    param_map = UniformParameterMap()
    assert param_map.premap_key == "params"
    assert param_map.projector.low == 0
    assert param_map.projector.high == 0
    assert param_map.projector.bound_lower == float("-inf")
    assert param_map.projector.bound_upper == float("+inf")
    # confirm that the tensors are unsqueezed to not be size 0
    assert param_map.projector.low.size() == (1,)
    assert param_map.projector.high.size() == (1,)
    assert param_map.projector.bound_lower.size() == (1,)
    assert param_map.projector.bound_upper.size() == (1,)

    # regular init
    param_map = UniformParameterMap(
        premap_key="params",
        tensor_low=torch.tensor([0.0, 0.0, 0.0]),
        tensor_high=torch.tensor([2.0, 2.0, 2.0]),
    )
    assert param_map.premap_key == "params"
    assert torch.allclose(param_map.projector.low, torch.tensor([0.0, 0.0, 0.0]))
    assert torch.allclose(param_map.projector.high, torch.tensor([2.0, 2.0, 2.0]))
    assert param_map.projector.bound_lower == float("-inf")
    assert param_map.projector.bound_upper == float("+inf")

    # float init
    param_map = UniformParameterMap(tensor_low=0.0, tensor_high=2.0)
    assert param_map.premap_key == "params"
    assert param_map.projector.low == 0
    assert param_map.projector.high == 2
    assert param_map.projector.bound_lower == float("-inf")
    assert param_map.projector.bound_upper == float("+inf")
    # confirm that the tensors are unsqueezed to not be size 0
    assert param_map.projector.low.size() == (1,)
    assert param_map.projector.high.size() == (1,)
    assert param_map.projector.bound_lower.size() == (1,)
    assert param_map.projector.bound_upper.size() == (1,)

    # array init
    param_map = UniformParameterMap(
        premap_key="params", tensor_low=[0.0, 0.0, 0.0], tensor_high=[2.0, 2.0, 2.0]
    )
    assert param_map.premap_key == "params"
    assert torch.allclose(param_map.projector.low, torch.tensor([0.0, 0.0, 0.0]))
    assert torch.allclose(param_map.projector.high, torch.tensor([2.0, 2.0, 2.0]))
    assert param_map.projector.bound_lower == float("-inf")
    assert param_map.projector.bound_upper == float("+inf")


def test_sample():
    param_map = UniformParameterMap(
        premap_key="params",
        tensor_low=torch.tensor([0.0, 0.0, 0.0]),
        tensor_high=torch.tensor([2.0, 2.0, 2.0]),
    )
    sample_tensor = param_map.sample()
    assert torch.all(torch.greater(sample_tensor, torch.tensor([0.0])))
    assert torch.all(torch.less(sample_tensor, torch.tensor([2.0])))


def test_map():
    input_dict = {"metadata": 1}
    param_map = UniformParameterMap(
        premap_key="params",
        tensor_low=torch.tensor([0.0, 0.0, 0.0]),
        tensor_high=torch.tensor([2.0, 2.0, 2.0]),
    )
    output_dict = param_map.map(input_dict)
    assert output_dict["params"].size()[0] == 3
    assert torch.all(torch.greater(output_dict["params"], param_map.projector.low))
    assert torch.all(torch.less(output_dict["params"], param_map.projector.high))

    # test override
    input_dict = {"metadata": 1, "params": torch.tensor([5.0, 5.0, 5.0])}
    param_map = UniformParameterMap(
        premap_key="params",
        tensor_low=torch.tensor([0.0, 0.0, 0.0]),
        tensor_high=torch.tensor([2.0, 2.0, 2.0]),
    )
    output_dict = param_map.map(input_dict, override_existing=True)
    assert not torch.allclose(output_dict["params"], torch.tensor([5.0, 5.0, 5.0]))

    # test no override
    input_dict = {"metadata": 1, "params": torch.tensor([5.0, 5.0, 5.0])}
    param_map = UniformParameterMap(
        premap_key="params",
        tensor_low=torch.tensor([0.0, 0.0, 0.0]),
        tensor_high=torch.tensor([2.0, 2.0, 2.0]),
    )
    output_dict = param_map.map(input_dict, override_existing=False)
    assert torch.allclose(output_dict["params"], torch.tensor([5.0, 5.0, 5.0]))
    assert torch.allclose(param_map.projector.high, torch.tensor([5.0, 5.0, 5.0]))


def test_save():
    input_dict = {"metadata": 1}
    param_map = UniformParameterMap(
        premap_key="params",
        tensor_low=torch.tensor([0.0, 0.0, 0.0]),
        tensor_high=torch.tensor([2.0, 2.0, 2.0]),
    )
    output_dict = param_map.map(input_dict)
    uid = param_map.save_leaf(resource_uri=RESOURCE_URI)

    new_map = UniformParameterMap()
    loaded_map = new_map.load_leaf(uid, resource_uri=RESOURCE_URI)
    assert torch.allclose(loaded_map.projector.high, torch.tensor([2.0, 2.0, 2.0]))
