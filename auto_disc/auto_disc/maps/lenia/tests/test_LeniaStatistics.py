from copy import deepcopy

import torch
from adtool_default.maps.LeniaStatistics import LeniaStatistics


def test_LeniaStatistics___init__():
    encoder = LeniaStatistics()


def test_LeniaStatistics_map():
    input_dict = {"output": torch.rand((64, 64))}
    original_input = deepcopy(input_dict)
    encoder = LeniaStatistics(SX=64, SY=64)
    output_dict = encoder.map(input_dict)

    assert "raw_output" in output_dict
    assert "output" in output_dict

    assert torch.allclose(original_input["output"], output_dict["raw_output"])
    assert output_dict["output"].size() == (17,)
