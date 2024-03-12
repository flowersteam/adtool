import os
from copy import deepcopy

import neat.genome
from adtool_default.maps.NEATParameterMap import NEATParameterMap


def setup_function(function):
    global CONFIG_PATH, CONFIG_STR
    # get current file path
    tests_path = os.path.dirname(os.path.abspath(__file__))
    maps_path = os.path.dirname(tests_path)
    CONFIG_PATH = os.path.join(maps_path, "cppn/config.cfg")
    with open(CONFIG_PATH, "r") as f:
        CONFIG_STR = f.read()

    return


def test_NEATParameterMap___init__():
    # initialize with config path
    NEATParameterMap(config_path=CONFIG_PATH)

    # initialize with config str
    NEATParameterMap(config_str=CONFIG_STR)


def test_NEATParameterMap_sample():
    # NOTE: This test is not deterministic.
    neat_map = NEATParameterMap(config_path=CONFIG_PATH)
    genome = neat_map.sample()

    assert isinstance(genome, neat.genome.DefaultGenome)


def test_NEATParameterMap_map():
    input_dict = {"random_data": 1, "metadata": "hello"}
    neat_map = NEATParameterMap(config_path=CONFIG_PATH)
    out = neat_map.map(input_dict)
    assert "genome" in out
    assert "neat_config" in out
    # NOTE: same memory object being pointed to
    assert out["neat_config"] is neat_map.neat_config

    dc = deepcopy(out)
    assert dc["neat_config"] is not neat_map.neat_config

    # TODO: postmap_shape = (1,1) edge case is broken
