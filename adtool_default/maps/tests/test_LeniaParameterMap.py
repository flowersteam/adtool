import os
from copy import deepcopy
from dataclasses import asdict

import pytest
import torch
from adtool_default.systems.Lenia import LeniaDynamicalParameters
from auto_disc.auto_disc.maps import NEATParameterMap
from auto_disc.auto_disc.maps.lenia.LeniaParameterMap import (
    LeniaHyperParameters,
    LeniaParameterMap,
)
from auto_disc.utils.leaf.locators.locators import BlobLocator


def setup_function(function):
    global CONFIG, CONFIG_PATH, CONFIG_STR
    path = os.path.dirname(os.path.abspath(__file__))
    CONFIG_PATH = os.path.join(path, "config_test.cfg")
    with open(CONFIG_PATH, "r") as f:
        CONFIG_STR = f.read()

    CONFIG = LeniaHyperParameters()


def test_LeniaHyperParamaters___init__():
    hp = LeniaHyperParameters()
    assert hp.tensor_low.size() == (8,)
    assert hp.tensor_high.size() == (8,)
    assert hp.tensor_bound_low.size() == (8,)
    assert hp.tensor_bound_high.size() == (8,)
    assert hp.init_state_dim == (10, 10)
    assert hp.cppn_n_passes == 2


def test_LeniaParamaterMap___init__():
    map = LeniaParameterMap(param_obj=CONFIG, neat_config_path=CONFIG_PATH)
    assert isinstance(map.locator, BlobLocator)

    # replace with kwargs
    with pytest.raises(Exception) as ex:
        map = LeniaParameterMap(param_obj=CONFIG, neat_config_path="test")
        assert "No such config" in ex
    map = LeniaParameterMap(
        param_obj=CONFIG, neat_config_path=CONFIG_PATH, cppn_n_passes=3
    )
    assert map.cppn_n_passes == 3

    # initialize with neat_config_path
    map = LeniaParameterMap(
        param_obj=CONFIG, neat_config_path=CONFIG_PATH, init_state_dim=(3, 3)
    )
    assert map.SX == 3
    assert map.SY == 3

    # initialize with neat_config_str
    map = LeniaParameterMap(
        param_obj=CONFIG, neat_config_str=CONFIG_STR, init_state_dim=(3, 3)
    )


def test_LeniaParameterMap_map():
    map = LeniaParameterMap(param_obj=CONFIG, neat_config_path=CONFIG_PATH)
    input = {}
    output = map.map(input)

    assert "params" in output
    assert "genome" in output["params"]
    assert "neat_config" in output["params"]
    assert "dynamic_params" in output["params"]


def test_LeniaParameterMap_sample():
    map = LeniaParameterMap(param_obj=CONFIG, neat_config_path=CONFIG_PATH)
    params_dict = map.sample()

    assert "genome" in params_dict
    assert "neat_config" in params_dict
    assert "dynamic_params" in params_dict


def test_LeniaParameterMap_mutate():
    # NOTE: not deterministic
    map = LeniaParameterMap(param_obj=CONFIG, neat_config_path=CONFIG_PATH)
    params_dict = map.sample()
    orig_dyn_p = LeniaDynamicalParameters(**params_dict["dynamic_params"]).to_tensor()
    genome = params_dict["genome"]
    orig_init_state = (
        map._cppn_map_genome(genome, map.neat.neat_config).detach().clone()
    )

    # mutation
    params_dict = map.mutate(params_dict)

    assert "genome" in params_dict
    assert "dynamic_params" in params_dict

    # check that the dynamic params have been mutated
    new_dyn_p = LeniaDynamicalParameters(**params_dict["dynamic_params"]).to_tensor()
    assert not torch.allclose(orig_dyn_p, new_dyn_p)

    # check that the genome has been mutated, leading to a new init_state
    genome = params_dict["genome"]
    new_init_state = map._cppn_map_genome(genome, map.neat.neat_config)
    assert not torch.allclose(orig_init_state, new_init_state)
