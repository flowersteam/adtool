from dataclasses import dataclass

import pytest
from auto_disc.auto_disc.utils.expose_config.defaults import (
    Defaults,
    deconstruct_recursive_dataclass_instance,
    defaults,
)
from auto_disc.auto_disc.utils.expose_config.expose_config import expose_config
import sys
class TestPublicExposeConfig:
    def test_key_collision(self):
        with pytest.raises(ValueError):

            @expose_config("a", default=False)
            @expose_config("a", default=False)
            class M:
                def __init__(self, a):
                    self.a = a

    def test_boolean(self):
        @expose_config("a", default=False)
        class M:
            def __init__(self, a):
                self.a = a

        assert M.CONFIG_DEFINITION["a"]["type"] == "BOOLEAN"
        assert M.CONFIG_DEFINITION["a"]["default"] == False

    def test_integer(self):
        @expose_config("a", default=2, domain=[1, 3])
        class M:
            def __init__(self, a):
                self.a = a

        assert M.CONFIG_DEFINITION["a"]["type"] == "INTEGER"
        assert M.CONFIG_DEFINITION["a"]["default"] == 2
        assert M.CONFIG_DEFINITION["a"]["min"] == 1
        assert M.CONFIG_DEFINITION["a"]["max"] == 3

    def test_float(self):
        @expose_config("a", default=2.0, domain=[1.0, 3.0])
        class M:
            def __init__(self, a):
                self.a = a

        assert M.CONFIG_DEFINITION["a"]["type"] == "DECIMAL"
        assert M.CONFIG_DEFINITION["a"]["default"] == 2.0
        assert M.CONFIG_DEFINITION["a"]["min"] == 1.0
        assert M.CONFIG_DEFINITION["a"]["max"] == 3.0

    def test_string(self):
        @expose_config("a", default="b", domain=["a", "b", "c"])
        class M:
            def __init__(self, a):
                self.a = a

        assert M.CONFIG_DEFINITION["a"]["type"] == "STRING"
        assert M.CONFIG_DEFINITION["a"]["default"] == "b"
        assert M.CONFIG_DEFINITION["a"]["possible_values"] == ["a", "b", "c"]

    def test_dict(self):
        @expose_config("a", default={"data": 1})
        class M:
            def __init__(self, a):
                self.a = a

        assert M.CONFIG_DEFINITION["a"]["type"] == "DICT"
        assert M.CONFIG_DEFINITION["a"]["default"] == {"data": 1}

    def test_submodule(self):
        class S:
            def __init__(self, b):
                self.b = b

        class S2:
            def __init__(self, d):
                self.d = d

        # note that the order of decorators is unimportant
        @expose_config("a", default=5, domain=[4, 5, 6])
        @expose_config("b", default=2, domain=[1, 2, 3], parent="S")
        @expose_config("c", default=8, domain=[7, 8, 9], parent="S")
        @expose_config("d", default=2, domain=[1, 2, 3], parent="S2")
        class M:
            def __init__(self, a, b, c, d):
                self.a = a
                self.sub = S(b, c)
                self.sub_again = S2(d)

        assert M.CONFIG_DEFINITION["a"]["type"] == "INTEGER"
        assert M.CONFIG_DEFINITION["a"]["default"] == 5
        assert M.CONFIG_DEFINITION["a"]["min"] == 4
        assert M.CONFIG_DEFINITION["a"]["max"] == 6
        assert M.CONFIG_DEFINITION["b"]["type"] == "INTEGER"
        assert M.CONFIG_DEFINITION["b"]["default"] == 2
        assert M.CONFIG_DEFINITION["b"]["min"] == 1
        assert M.CONFIG_DEFINITION["b"]["max"] == 3
        assert M.CONFIG_DEFINITION["b"]["parent"] == "S"
        assert M.CONFIG_DEFINITION["c"]["type"] == "INTEGER"
        assert M.CONFIG_DEFINITION["c"]["default"] == 8
        assert M.CONFIG_DEFINITION["c"]["min"] == 7
        assert M.CONFIG_DEFINITION["c"]["max"] == 9
        assert M.CONFIG_DEFINITION["c"]["parent"] == "S"
        assert M.CONFIG_DEFINITION["d"]["type"] == "INTEGER"
        assert M.CONFIG_DEFINITION["d"]["default"] == 2
        assert M.CONFIG_DEFINITION["d"]["min"] == 1
        assert M.CONFIG_DEFINITION["d"]["max"] == 3
        assert M.CONFIG_DEFINITION["d"]["parent"] == "S2"


class TestComplicatedExposeConfig:
    def test_dataclass_expose(self):
        @dataclass(frozen=True)
        class SystemParams(Defaults):
            version: str = defaults("fft", domain=["fft", "conv"])
            SX: int = defaults(256, min=1, max=2048)
            SY: int = defaults(256, min=1, max=2048)
            final_step: int = defaults(200, min=1, max=1000)
            scale_init_state: float = defaults(1.0, domain=[1.0, 100.0])

        # I don't think can remove the unaesthetic () at the end of this method
        # unless we use metaclasses which I don't want to do - Jesse
        @SystemParams.expose_config()
        class System:
            def __init__(self, *args, **kwargs):
                pass

        assert System.CONFIG_DEFINITION["version"]["type"] == "STRING"
        assert System.CONFIG_DEFINITION["version"]["default"] == "fft"
        assert System.CONFIG_DEFINITION["version"]["possible_values"] == ["fft", "conv"]
        assert System.CONFIG_DEFINITION["version"]["parent"] == ""

        assert System.CONFIG_DEFINITION["SX"]["type"] == "INTEGER"
        assert System.CONFIG_DEFINITION["SX"]["default"] == 256
        assert System.CONFIG_DEFINITION["SX"]["min"] == 1
        assert System.CONFIG_DEFINITION["SX"]["max"] == 2048
        assert System.CONFIG_DEFINITION["SX"]["parent"] == ""

        assert System.CONFIG_DEFINITION["SY"]["type"] == "INTEGER"
        assert System.CONFIG_DEFINITION["SY"]["default"] == 256
        assert System.CONFIG_DEFINITION["SY"]["min"] == 1
        assert System.CONFIG_DEFINITION["SY"]["max"] == 2048
        assert System.CONFIG_DEFINITION["SY"]["parent"] == ""

        assert System.CONFIG_DEFINITION["final_step"]["type"] == "INTEGER"
        assert System.CONFIG_DEFINITION["final_step"]["default"] == 200
        assert System.CONFIG_DEFINITION["final_step"]["min"] == 1
        assert System.CONFIG_DEFINITION["final_step"]["max"] == 1000
        assert System.CONFIG_DEFINITION["final_step"]["parent"] == ""

        assert System.CONFIG_DEFINITION["scale_init_state"]["type"] == "DECIMAL"
        assert System.CONFIG_DEFINITION["scale_init_state"]["default"] == 1.0
        assert System.CONFIG_DEFINITION["scale_init_state"]["min"] == 1.0
        assert System.CONFIG_DEFINITION["scale_init_state"]["max"] == 100.0
        assert System.CONFIG_DEFINITION["scale_init_state"]["parent"] == ""

    def test_expose_recursive(self):
        @dataclass(frozen=True)
        class Scalar(Defaults):
            scalar: float = defaults(1.0, domain=[1.0, 100.0])

        @dataclass(frozen=True)
        class Geometry(Defaults):
            SX: int = defaults(256, min=1, max=2048)
            SY: int = defaults(256, min=1, max=2048)
            scale_init_state: Scalar = Scalar()

        @dataclass(frozen=True)
        class SystemParams(Defaults):
            version: str = defaults("fft", domain=["fft", "conv"])
            size: Geometry = Geometry()

        # I don't think can remove the unaesthetic () at the end of this method
        # unless we use metaclasses which I don't want to do - Jesse
        @SystemParams.expose_config()
        class System:
            def __init__(self, *args, **kwargs):
                pass

        assert System.CONFIG_DEFINITION["version"]["type"] == "STRING"
        assert System.CONFIG_DEFINITION["version"]["default"] == "fft"
        assert System.CONFIG_DEFINITION["version"]["possible_values"] == ["fft", "conv"]

        assert System.CONFIG_DEFINITION["SX"]["type"] == "INTEGER"
        assert System.CONFIG_DEFINITION["SX"]["default"] == 256
        assert System.CONFIG_DEFINITION["SX"]["min"] == 1
        assert System.CONFIG_DEFINITION["SX"]["max"] == 2048
        assert System.CONFIG_DEFINITION["SX"]["parent"] == "size"

        assert System.CONFIG_DEFINITION["SY"]["type"] == "INTEGER"
        assert System.CONFIG_DEFINITION["SY"]["default"] == 256
        assert System.CONFIG_DEFINITION["SY"]["min"] == 1
        assert System.CONFIG_DEFINITION["SY"]["max"] == 2048
        assert System.CONFIG_DEFINITION["SY"]["parent"] == "size"

        assert System.CONFIG_DEFINITION["scalar"]["type"] == "DECIMAL"
        assert System.CONFIG_DEFINITION["scalar"]["default"] == 1.0
        assert System.CONFIG_DEFINITION["scalar"]["min"] == 1.0
        assert System.CONFIG_DEFINITION["scalar"]["max"] == 100.0
        assert System.CONFIG_DEFINITION["scalar"]["parent"] == "size.scale_init_state"

    def test_expose_recursive_key_cornercase(self):
        @dataclass(frozen=True)
        class Scalar(Defaults):
            # this does not cause a corner case, as the "size" attribute
            # in SystemParams causes a recursion and does not get added
            # to CONFIG_DEFINITION directly
            # THIS IS STILL UNADVISED
            size: float = defaults(1.0, domain=[1.0, 100.0])

        @dataclass(frozen=True)
        class Geometry(Defaults):
            SX: int = defaults(256, min=1, max=2048)
            SY: int = defaults(256, min=1, max=2048)
            scale_init_state: Scalar = Scalar()

        @dataclass(frozen=True)
        class SystemParams(Defaults):
            version: str = defaults("fft", domain=["fft", "conv"])
            size: Geometry = Geometry()

        @SystemParams.expose_config()
        class System:
            def __init__(self, *args, **kwargs):
                pass

    def test_expose_recursive_key_collision(self):
        @dataclass(frozen=True)
        class Scalar(Defaults):
            # this causes the key collision
            version: str = defaults("float", domain=["float", "int"])
            scalar: float = defaults(1.0, domain=[1.0, 100.0])

        @dataclass(frozen=True)
        class Geometry(Defaults):
            SX: int = defaults(256, min=1, max=2048)
            SY: int = defaults(256, min=1, max=2048)
            scale_init_state: Scalar = Scalar()

        @dataclass(frozen=True)
        class SystemParams(Defaults):
            version: str = defaults("fft", domain=["fft", "conv"])
            size: Geometry = Geometry()

        with pytest.raises(ValueError) as e:

            @SystemParams.expose_config()
            class System:
                def __init__(self, *args, **kwargs):
                    pass

        assert "already exists" in str(e.value)


class TestInitDecoratedObject:
    def test_no_magic(self):
        @dataclass(frozen=True)
        class Scalar(Defaults):
            scalar: float = defaults(1.0, domain=[1.0, 100.0])

        @dataclass(frozen=True)
        class Geometry(Defaults):
            SX: int = defaults(256, min=1, max=2048)
            SY: int = defaults(256, min=1, max=2048)
            scale_init_state: Scalar = Scalar()

        @dataclass(frozen=True)
        class SystemParams(Defaults):
            version: str = defaults("fft", domain=["fft", "conv"])
            size: Geometry = Geometry()

        @SystemParams.expose_config()
        class System:
            def __init__(self):
                pass

        # unlike the old ConfigParameter decorators, this one does not magically
        # change the __init__ function
        with pytest.raises(TypeError) as e:
            System(version="conv", scalar=1)
        assert "argument" in str(e.value)
        assert not System.__dict__.get("config", None)
        assert System.CONFIG_DEFINITION

    def test_init_structured(self):
        @dataclass(frozen=True)
        class Scalar(Defaults):
            scalar: float = defaults(1.0, domain=[1.0, 100.0])

        @dataclass(frozen=True)
        class Geometry(Defaults):
            SX: int = defaults(256, min=1, max=2048)
            SY: int = defaults(256, min=1, max=2048)
            scale_init_state: Scalar = Scalar()

        @dataclass(frozen=True)
        class SystemParams(Defaults):
            version: str = defaults("fft", domain=["fft", "conv"])
            size: Geometry = Geometry()

        @SystemParams.expose_config()
        class System:
            def __init__(self, params: SystemParams):
                self.version = params.version

        p = SystemParams(
            version="conv", size=Geometry(SX=100, SY=100, scale_init_state=Scalar(1))
        )
        s = System(params=p)
        assert s.version == "conv"

    def test_init_destructured(self):
        # this is the default way to initialize, which is supported by
        # ExperimentalPipeline

        @dataclass(frozen=True)
        class Scalar(Defaults):
            scalar: float = defaults(1.0, domain=[1.0, 100.0])

        @dataclass(frozen=True)
        class Geometry(Defaults):
            SX: int = defaults(256, min=1, max=2048)
            SY: int = defaults(256, min=1, max=2048)
            scale_init_state: Scalar = Scalar()

        @dataclass(frozen=True)
        class SystemParams(Defaults):
            version: str = defaults("fft", domain=["fft", "conv"])
            size: Geometry = Geometry()

        @SystemParams.expose_config()
        class System:
            def __init__(self, version: str, SX: int, SY: int, scalar: float):
                self.version = version
                self.SX = SX
                self.SY = SY
                self.scalar = scalar

        p = SystemParams()
        p_dict = deconstruct_recursive_dataclass_instance(p)
        assert System(**p_dict).SX == 256
        assert System(**p_dict).SY == 256
        assert System(**p_dict).version == "fft"
        assert System(**p_dict).scalar == 1.0

        # similar test
        @SystemParams.expose_config()
        class System:
            def __init__(self, **params):
                self.version = params["version"]
                self.SX = params["SX"]
                self.SY = params["SY"]
                self.scalar = params["scalar"]

        p = SystemParams()
        p_dict = deconstruct_recursive_dataclass_instance(p)
        assert System(**p_dict).SX == 256
        assert System(**p_dict).SY == 256
        assert System(**p_dict).version == "fft"
        assert System(**p_dict).scalar == 1.0


class TestMultipleClasses:
    def test_basic(self):
        @dataclass(frozen=True)
        class SystemParams(Defaults):
            version: str = defaults("fft", domain=["fft", "conv"])

        @SystemParams.expose_config()
        class System:
            def __init__(self, version: str):
                self.version = version

        @dataclass(frozen=True)
        class OtherParams(Defaults):
            other_version: str = defaults("fft", domain=["fft", "conv"])

        @OtherParams.expose_config()
        class OtherSystem:
            def __init__(self, other_version: str):
                self.other_version = other_version

        # assert OtherSystem.CONFIG_DEFINITION != System.CONFIG_DEFINITION
        # assert OtherSystem.CONFIG_DEFINITION["other_version"] is not None
        # assert System.CONFIG_DEFINITION["version"] is not None

    def test_inheritance(self):
        class Base:
            CONFIG_DEFINITION = {}

        @dataclass(frozen=True)
        class SystemParams(Defaults):
            version: str = defaults("fft", domain=["fft", "conv"])

        
        @SystemParams.expose_config()
        class System(Base):
            def __init__(self, version: str):
                self.version = version


        @dataclass(frozen=True)
        class OtherParams(Defaults):
            other_version: str = defaults("fft", domain=["fft", "conv"])

        @OtherParams.expose_config()
        class OtherSystem(Base):
            def __init__(self, other_version: str):
                self.other_version = other_version


        assert OtherSystem.CONFIG_DEFINITION != System.CONFIG_DEFINITION
                

        assert OtherSystem.CONFIG_DEFINITION["other_version"] is not None
        assert System.CONFIG_DEFINITION["version"] is not None
