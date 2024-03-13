import json
import os
import pathlib
import pickle
import shutil
from hashlib import sha1

import auto_disc.auto_disc
import auto_disc.utils.leaf
import numpy
import pytest
import torch
from auto_disc.legacy.utils.callbacks.on_discovery_callbacks.save_discovery_on_disk import (
    SaveDiscoveryOnDisk,
)
from auto_disc.utils.leaf.Leaf import Leaf
from pytest_mock import mocker


def setup_function(function):
    global RESOURCE_URI
    file_path = str(pathlib.Path(__file__).parent.resolve())
    RESOURCE_URI = os.path.join(file_path, "tmp")
    os.mkdir(RESOURCE_URI)
    return


def teardown_function(function):
    global RESOURCE_URI
    if os.path.exists(RESOURCE_URI):
        shutil.rmtree(RESOURCE_URI)
    return


def test_SaveDiscoveryOnDisk__construct_save_path():
    cb = SaveDiscoveryOnDisk()
    dir_path = cb._initialize_save_path(RESOURCE_URI, "test_id", 777, 33)

    assert os.path.dirname(dir_path) == RESOURCE_URI + "/discoveries"


def test__save_binary_callback():
    cb = SaveDiscoveryOnDisk()
    dir_path = cb._initialize_save_path(RESOURCE_URI, "test_id", 777, 33)
    bin = b"123"
    uid = cb._save_binary_callback(bin, dir_path)
    assert os.path.exists(os.path.join(dir_path, uid))


def test___call__():
    data = {
        "params": torch.tensor([1.0, 2.0, 3.0]),
        "loss": 0.5,
        "model": object(),
        "metadata": {"test": b"test"},
    }
    cb = SaveDiscoveryOnDisk()
    cb(
        resource_uri=RESOURCE_URI,
        experiment_id="test_id",
        seed=33,
        run_idx=777,
        discovery=data,
    )

    # check if the discovery was saved
    dir_path = cb._initialize_save_path(RESOURCE_URI, "test_id", 777, 33)
    assert os.path.exists(dir_path)
    assert os.path.exists(os.path.join(dir_path, "discovery.json"))

    # load the discovery and check if the data is correct
    obj_hash = sha1(pickle.dumps(data["model"])).hexdigest()
    byte_hash = sha1(data["metadata"]["test"]).hexdigest()

    with open(os.path.join(dir_path, "discovery.json"), "r") as f:
        loaded_data = json.load(f)
    assert loaded_data["params"] == [1.0, 2.0, 3.0]
    assert loaded_data["loss"] == 0.5
    assert loaded_data["model"] == obj_hash
    assert loaded_data["metadata"] == {"test": byte_hash}

    with open(os.path.join(dir_path, obj_hash), "rb") as f:
        loaded_data = f.read()
    assert loaded_data == pickle.dumps(data["model"])
    with open(os.path.join(dir_path, byte_hash), "rb") as f:
        loaded_data = f.read()
    assert loaded_data == b"test"
