import pytest_mock
import torch
from auto_disc.legacy.utils.callbacks.on_discovery_callbacks.save_discovery_in_expedb import (
    SaveDiscoveryInExpeDB,
)


def test___call__(mocker):
    data = {
        "params": torch.tensor([1.0, 2.0, 3.0]),
        "loss": 0.5,
        "model": object(),
        "metadata": {"test": b"test"},
    }
    resource_uri = "http://127.0.0.1:5001"
    cb = SaveDiscoveryInExpeDB()
    spy = mocker.spy(cb, "_initialize_save_path")
    cb(
        resource_uri=resource_uri,
        experiment_id="test_id",
        run_idx=777,
        seed=33,
        discovery=data,
    )

    # TODO: check if the discovery was saved by interfacing with the MongoDB
    # for now, just use this spy and manually check the DB
    assert spy.spy_return
