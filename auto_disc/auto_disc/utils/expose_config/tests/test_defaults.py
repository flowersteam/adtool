#!/usr/bin/env python3
from dataclasses import dataclass

import pytest
from auto_disc.auto_disc.utils.expose_config.defaults import (
    Defaults,
    _is_default_dataclass,
    _is_default_field,
    _is_default_field_r,
    deconstruct_recursive_dataclass_instance,
    defaults,
)


@dataclass
class Scalar(Defaults):
    # this does not cause a corner case, as the "size" attribute
    # in SystemParams causes a recursion and does not get added
    # to CONFIG_DEFINITION directly
    # THIS IS STILL UNADVISED
    size: float = defaults(1.0, domain=[1.0, 100.0])


@dataclass
class Geometry(Defaults):
    SX: int = defaults(256, min=1, max=2048)
    SY: int = defaults(256, min=1, max=2048)
    scale_init_state: Scalar = Scalar()


@dataclass
class SystemParams(Defaults):
    version: str = defaults("fft", domain=["fft", "conv"])
    size: Geometry = Geometry()


def test__is_default_dataclass():
    assert _is_default_dataclass(Scalar())
    assert not _is_default_dataclass(Scalar(size=2))
    assert _is_default_dataclass(Geometry())
    # note: initializing to the same value as the default counts as initializing!
    assert not _is_default_dataclass(Geometry(scale_init_state=Scalar(size=1.0)))
    # note: this edge case is unrealistic and redundant
    assert _is_default_dataclass(Geometry(scale_init_state=Scalar()))


def test__is_default_field():
    assert _is_default_field(Geometry(), "SX")
    assert _is_default_field(Geometry(), "scale_init_state")
    assert not _is_default_field(Geometry(SX=1), "SX")
    # note: initializing to the same value as the default counts as initializing!
    assert not _is_default_field(Geometry(SX=256), "SX")


def test__is_default_field_r():
    assert _is_default_field_r(SystemParams(), "version")
    assert _is_default_field_r(SystemParams(), "scale_init_state")
    assert not _is_default_field_r(SystemParams(size=Geometry(SX=1)), "SX")
    assert not _is_default_field_r(SystemParams(size=Geometry(SX=1)), "size")

    # note: initializing to the same value as the default counts as initializing!
    assert not _is_default_field_r(SystemParams(size=Geometry(SX=256)), "size")
    assert not _is_default_field_r(SystemParams(size=Geometry(SX=256)), "SX")

    # some extras for stress testing
    assert _is_default_field_r(SystemParams(size=Geometry(SX=1)), "scale_init_state")
    assert not _is_default_field_r(
        SystemParams(size=Geometry(scale_init_state=Scalar(size=1))), "scale_init_state"
    )
    assert _is_default_field_r(
        SystemParams(size=Geometry(scale_init_state=Scalar())), "scale_init_state"
    )


def test_deconstruct_dataclass_instance():
    flat_config_dict = {"version": "fft", "SX": 256, "SY": 256, "size": 1.0}
    assert flat_config_dict == deconstruct_recursive_dataclass_instance(SystemParams())

    flat_config_dict["size"] = 2
    assert flat_config_dict == deconstruct_recursive_dataclass_instance(
        SystemParams(size=Geometry(scale_init_state=Scalar(size=2)))
    )

    # basic example of key name clobbering which works fine, because
    # the `version` keyname is a nested dataclass, and dataclasses are destructured
    @dataclass
    class BadGeometry(Defaults):
        SX: int = defaults(256, min=1, max=2048)
        SY: int = defaults(256, min=1, max=2048)
        version: Scalar = Scalar()

    @dataclass
    class BadParams(Defaults):
        version: str = defaults("fft", domain=["fft", "conv"])
        size: BadGeometry = BadGeometry()

    assert flat_config_dict == deconstruct_recursive_dataclass_instance(
        BadParams(size=BadGeometry(version=Scalar(size=2)))
    )

    # basic example of key name clobbering which raises exception
    @dataclass
    class BadGeometry(Defaults):
        version: int = defaults(256, min=1, max=2048)
        SY: int = defaults(256, min=1, max=2048)
        scale_init_state: Scalar = Scalar()

    @dataclass
    class BadParams(Defaults):
        version: str = defaults("fft", domain=["fft", "conv"])
        size: BadGeometry = BadGeometry()

    with pytest.raises(KeyError):
        deconstruct_recursive_dataclass_instance(
            BadParams(size=BadGeometry(version=Scalar(size=2)))
        )
