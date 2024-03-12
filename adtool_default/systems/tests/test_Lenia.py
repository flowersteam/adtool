from copy import deepcopy
from dataclasses import asdict

import torch
from adtool_default.systems.Lenia import (
    Lenia,
    LeniaDynamicalParameters,
    LeniaParameters,
)
from auto_disc.utils.filetype_converter.filetype_converter import is_mp4


def setup_function(function):
    global dummy_input
    dummy_input = {
        "params": {
            "dynamic_params": {
                "R": torch.tensor(5.0),
                "T": torch.tensor(10.0),
                "b": torch.tensor([0.1, 0.2, 0.3, 0.4]),
                "m": torch.tensor(0.5),
                "s": torch.tensor(0.1),
            },
            "init_state": torch.rand((256, 256)),
        }
    }


def teardown_function(function):
    pass


def test_LeniaDynamicalParamaters___init__():
    p = LeniaDynamicalParameters()
    assert p.R == 0
    assert p.T == 1.0
    assert p.b.size() == (4,)
    assert isinstance(p.b, torch.Tensor)
    assert p.m == 0.0
    assert p.s == 0.001

    # check custom initialization
    p = LeniaDynamicalParameters(
        R=10, T=2.0, b=torch.tensor([1.0, 1.0, 1.0, 1.0]), m=0.5, s=0.2
    )
    assert p.R == 10
    assert p.T == 2.0
    assert p.b.size() == (4,)
    assert isinstance(p.b, torch.Tensor)
    assert p.m == 0.5
    assert p.s == 0.2

    # check rounding of R
    p = LeniaDynamicalParameters(R=5.5)
    assert p.R == 6


def test_LeniaDynamicalParamaters___post_init__():
    # assert that all constraints are respected
    p = LeniaDynamicalParameters(
        R=20, T=11.0, b=torch.tensor([2.0, -1, 1.0, 1.0]), m=1.5, s=0.5
    )
    assert p.R == 19
    assert p.T == 10.0
    assert torch.max(p.b) == 1.0
    assert torch.min(p.b) == 0.0
    assert torch.allclose(p.b, torch.tensor([1.0, 0.0, 1.0, 1.0]))
    assert p.m == 1.0
    assert p.s == 0.3


def test_LeniaDynamicalParamaters_to_tensor():
    p = LeniaDynamicalParameters()
    tensor = p.to_tensor()
    assert tensor.size() == (8,)
    assert isinstance(tensor, torch.Tensor)
    assert tensor[0] == p.R
    assert tensor[1] == p.T
    assert tensor[2] == p.m
    assert tensor[3] == p.s
    assert torch.allclose(tensor[4:8], p.b)


def test_LeniaParameters___init__():
    p = LeniaParameters()
    assert p.init_state.size() == (10, 10)
    assert isinstance(p.dynamic_params, LeniaDynamicalParameters)

    # check dict rep
    pdict = asdict(p)
    assert pdict["init_state"].size() == (10, 10)
    assert "R" in pdict["dynamic_params"]
    assert "T" in pdict["dynamic_params"]
    assert "b" in pdict["dynamic_params"]
    assert "m" in pdict["dynamic_params"]
    assert "s" in pdict["dynamic_params"]


def test_Lenia___init__():
    system = Lenia()
    assert system.orbit.size() == (200, 1, 1, 256, 256)


def test_Lenia_process_dict():
    system = Lenia()
    dummy_params = system._process_dict(dummy_input)
    assert dummy_params.dynamic_params.R


def test_Lenia__generate_automaton():
    system = Lenia()
    dummy_params = system._process_dict(dummy_input)

    automaton = system._generate_automaton(dummy_params.dynamic_params)
    assert isinstance(automaton, torch.nn.Module)


def test_Lenia__bootstrap():
    system = Lenia()
    dummy_params = system._process_dict(dummy_input)
    system._bootstrap(dummy_params)
    init_state = system.orbit[0]
    assert init_state.size() == (1, 1, 256, 256)


def test_Lenia__step():
    system = Lenia()
    dummy_params = system._process_dict(dummy_input)
    system._bootstrap(dummy_params)
    init_state = system.orbit[0]
    automaton = system._generate_automaton(dummy_params.dynamic_params)
    new_state = system._step(init_state, automaton)

    assert not torch.allclose(new_state, init_state)
    assert automaton.T.grad is None
    assert automaton.m.grad is None
    assert automaton.s.grad is None

    # generate ad graph
    new_state.sum().backward()
    assert automaton.T.grad is not None
    assert automaton.m.grad is not None
    assert automaton.s.grad is not None


def test_Lenia_map():
    system = Lenia()
    out_dict = system.map(dummy_input)

    assert torch.allclose(out_dict["output"], system.orbit[-1])
    assert out_dict["output"].size() == (256, 256)
    # padded due to the way automaton steps
    assert system.orbit[-1].size() == (1, 1, 256, 256)


def test_Lenia_render():
    # eyeball test this one
    system = Lenia()
    dummy_params = system._process_dict(dummy_input)

    out_dict = system.map(dummy_input)

    imagebytes = system.render(out_dict)
    assert is_mp4(imagebytes)
