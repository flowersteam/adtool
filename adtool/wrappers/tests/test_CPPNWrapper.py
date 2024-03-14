import os
from copy import deepcopy

import neat.genome
import torch
from examples.maps.NEATParameterMap import NEATParameterMap
from adtool.wrappers.CPPNWrapper import CPPNWrapper


def setup_function(function):
    global CONFIG_PATH
    # get current file path
    tests_path = os.path.dirname(os.path.abspath(__file__))
    CONFIG_PATH = os.path.join(tests_path, "config_test.cfg")


def test_CPPNWrapper___init__():
    cppn_wrapper = CPPNWrapper()


def test_CPPNWrapper_map():
    input_dict = {"random_data": 1, "metadata": "hello"}
    neat_map = NEATParameterMap(config_path=CONFIG_PATH)
    param_dict = neat_map.map(input_dict)

    cppn_wrapper = CPPNWrapper()
    out = cppn_wrapper.map(param_dict)

    assert "init_state" in out
    assert out["init_state"].size() == (10, 10)
