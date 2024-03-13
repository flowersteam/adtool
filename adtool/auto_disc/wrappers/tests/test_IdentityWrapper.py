from typing import Tuple

from adtool.auto_disc.wrappers.IdentityWrapper import IdentityWrapper


def test__init__():
    id = IdentityWrapper()
    assert isinstance(id, IdentityWrapper)


def test_map():
    input = {"a": 1, "b": 2}
    id = IdentityWrapper()
    output = id.map(input)
    assert output == input


def test_save_leaf_load_leaf():
    id = IdentityWrapper()
    leaf_uid = id.save_leaf()

    id2 = id.load_leaf(leaf_uid)

    assert id.locator.resource_uri == id2.locator.resource_uri
    del id.locator
    del id2.locator
    assert id.__dict__ == id2.__dict__
