from adtool.wrappers.TransformWrapper import TransformWrapper
from adtool.utils.leaf.tests.test_leaf import DummyLocator


def test__init__():
    id = TransformWrapper()
    assert isinstance(id, TransformWrapper)


def test_map():
    input = {"a": 1, "b": 2}
    tw = TransformWrapper(premap_keys=["a", "b"], postmap_keys=["b", "a"])
    output = tw.map(input)
    assert output == {"a": 2, "b": 1}
    assert input == {"a": 1, "b": 2}


def test_map_missing():
    input = {"data": 1}
    tw = TransformWrapper(premap_keys=["output"], postmap_keys=["data"])
    output = tw.map(input)
    assert output == input


def test_saveload():
    tw = TransformWrapper(premap_keys=["a", "b"], postmap_keys=["b", "a"])

    db = {}
    tw.locator = DummyLocator(db)
    leaf_uid = tw.save_leaf()

    tw2 = tw.load_leaf(leaf_uid)

    del tw.locator
    del tw2.locator
    assert tw.__dict__ == tw2.__dict__
