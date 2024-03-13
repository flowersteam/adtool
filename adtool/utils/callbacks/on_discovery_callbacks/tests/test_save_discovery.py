import json
import unittest.mock as mock

import auto_disc.auto_disc
import auto_disc.utils.leaf
import numpy
import torch
from auto_disc.legacy.utils.callbacks.on_discovery_callbacks.save_discovery import (
    _CustomJSONEncoder,
    _JSONEncoderFactory,
)
from auto_disc.utils.leaf.Leaf import Leaf


def test__JSONEncoderFactory():
    def dummy_cb(x):
        return x

    fac = _JSONEncoderFactory()
    cls = fac(dir_path="dummy_path", custom_callback=dummy_cb)
    assert cls._dir_path == "dummy_path"
    assert cls._custom_callback == dummy_cb
    assert cls.__name__ == "_CustomJSONEncoder"
    assert _CustomJSONEncoder._dir_path == "dummy_path"
    assert _CustomJSONEncoder._custom_callback == dummy_cb


def test__CustomJSONEncoder(mocker):
    fac = _JSONEncoderFactory()
    mocked_cb = mock.Mock(return_value="dummy_uid")
    cls = fac(dir_path="dummy_path", custom_callback=mocked_cb)
    encoder = cls()

    # catch torch Tensors
    obj = torch.Tensor([1, 2, 3])
    encoded = encoder.default(obj)
    assert encoded == [1, 2, 3]

    # catch numpy arrays
    obj = numpy.array([1, 2, 3])
    encoded = encoder.default(obj)
    assert encoded == [1, 2, 3]

    # catch bytes
    obj = b"dummy_bytes"
    encoded = encoder.default(obj)
    assert encoded == "dummy_uid"
    assert mocked_cb.call_count == 1

    # catch Leaf objects
    obj = Leaf()
    mocker.patch("auto_disc.utils.leaf.Leaf.Leaf.save_leaf", return_value="dummy_uid")
    encoded = encoder.default(obj)
    assert encoded == "dummy_uid"
    assert auto_disc.utils.leaf.Leaf.Leaf.save_leaf.call_count == 1

    # catch python objects not serializable by JSON
    obj = object()
    encoded = encoder.default(obj)
    assert encoded == "dummy_uid"
    assert mocked_cb.call_count == 2

    # ensure that the default method is called
    obj = "dummy_string"
    mocker.patch("json.JSONEncoder.default", return_value="dummy_return")
    encoded = encoder.default(obj)
    assert encoded == "dummy_return"
    # it is called twice, because the first is in the try block
    assert json.JSONEncoder.default.call_count == 2
