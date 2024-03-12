import os
from copy import deepcopy

import torch
from adtool_default.systems.LeniaCPPN import LeniaCPPN
from auto_disc.auto_disc.maps.NEATParameterMap import NEATParameterMap
from auto_disc.utils.filetype_converter.filetype_converter import is_mp4


def setup_function(function):
    global dummy_input

    tests_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(tests_path, "config_test.cfg")

    dummy_input = {}
    neat_map = NEATParameterMap(config_path=config_path)
    genome_dict = neat_map.map(dummy_input)

    dummy_input = {
        "params": {
            "dynamic_params": {
                "R": torch.tensor(5.0),
                "T": torch.tensor(10.0),
                "b": torch.tensor([0.1, 0.2, 0.3, 0.4]),
                "m": torch.tensor(0.5),
                "s": torch.tensor(0.1),
            },
            "genome": genome_dict["genome"],
            "neat_config": genome_dict["neat_config"],
        }
    }


def teardown_function(function):
    pass


def test___init__():
    system = LeniaCPPN()


def test_map():
    system = LeniaCPPN()
    out = system.map(dummy_input)

    assert "output" in out
    assert out["output"].size() == (256, 256)
    assert system.lenia.orbit[-1].size() == (1, 1, 256, 256)
    assert out["output"] is not system.lenia.orbit[-1][0, 0]

    # ensure all parts of parameters are kept
    assert "params" in out
    assert "dynamic_params" in out["params"]
    assert "genome" in out["params"]
    assert "neat_config" in out["params"]

    # eyeball test the render
    imagebytes = system.render(out)
    assert is_mp4(imagebytes)
