from auto_disc.utils.leaf.Leaf import Leaf
from auto_disc.utils.leaf.tests.test_leaf import DummyLocator, DummyModule
from auto_disc.utils.leafutils.leafstructs.service import provide_leaf_as_service


class PlusOffset(Leaf):
    def __init__(self, offset):
        super().__init__()
        self.offset = offset

    def forward(self, x):
        return x + self.offset


def setup_function(function):
    global plusone, DB
    plusone = PlusOffset(1)
    DB = {}


def test_init():
    new_plusone = provide_leaf_as_service(plusone, DummyModule)
    assert new_plusone


def test_save_load():
    overloaded_plusone = provide_leaf_as_service(plusone, DummyModule)
    saved_uid = overloaded_plusone.save_leaf(DB)
    overloaded_plusone.offset = 2

    new_plusone = overloaded_plusone.load_leaf(saved_uid, DB)
    assert new_plusone.offset == 1
    assert new_plusone.forward(2) == 3
