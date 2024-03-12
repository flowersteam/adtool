import os
import pathlib
import shutil
from copy import deepcopy
from typing import Dict

import pytest
from auto_disc.auto_disc.wrappers import SaveWrapper, WrapperPipeline
from auto_disc.utils.leaf.Leaf import Leaf, StatelessLocator
from auto_disc.utils.leaf.locators.Locator import FileLocator
from auto_disc.utils.leaf.tests.test_leaf import DummyLocator


class TestWrapper(Leaf):
    def __init__(self, premap_key: str = "in", offset: int = 1) -> None:
        super().__init__()
        self.premap_key = premap_key
        self.offset = offset

    def map(self, input: Dict) -> Dict:
        output = deepcopy(input)
        output[self.premap_key] += self.offset
        return output


def test___init__():
    input = {"in": 1}
    a = TestWrapper(offset=1, premap_key="in")
    b = TestWrapper(offset=2, premap_key="in")
    wrapper_list = [a, b]
    all_wrappers = WrapperPipeline(wrappers=wrapper_list)
    assert all_wrappers.map(input) == b.map(a.map(input))
    assert all_wrappers.wrappers[0]._container_ptr == all_wrappers
    assert all_wrappers.wrappers[1]._container_ptr == all_wrappers
    assert all_wrappers.wrappers[0].name == "0"
    assert all_wrappers.wrappers[1].name == "1"


def test___init__mutually_exclusive():
    storage_db = {}
    input = {"in": 1}
    a = TestWrapper(offset=1, premap_key="in")
    b = TestWrapper(offset=2, premap_key="in")
    wrapper_list = [a, b]
    with pytest.raises(ValueError):
        all_wrappers = WrapperPipeline(
            wrappers=wrapper_list,
            locator=DummyLocator(storage_db),
            resource_uri=storage_db,
        )


def test___init__pass_locator():
    storage_db = {}
    input = {"in": 1}
    a = TestWrapper(offset=1, premap_key="in")
    b = TestWrapper(offset=2, premap_key="in")
    wrapper_list = [a, b]
    all_wrappers = WrapperPipeline(
        wrappers=wrapper_list, locator=DummyLocator(storage_db)
    )
    assert all_wrappers.wrappers[0].locator.resource_uri == storage_db
    assert all_wrappers.wrappers[1].locator.resource_uri == storage_db
    assert isinstance(all_wrappers.wrappers[0].locator, DummyLocator)
    assert isinstance(all_wrappers.wrappers[1].locator, DummyLocator)
    assert isinstance(all_wrappers.locator, DummyLocator)
    assert all_wrappers.locator.resource_uri == storage_db


def test___init__pass_uri():
    input = {"in": 1}
    a = TestWrapper(offset=1, premap_key="in")
    b = TestWrapper(offset=2, premap_key="in")
    a.locator = DummyLocator("")
    b.locator = DummyLocator("")
    wrapper_list = [a, b]
    all_wrappers = WrapperPipeline(wrappers=wrapper_list, resource_uri="com.test")
    assert all_wrappers.wrappers[0].locator.resource_uri == "com.test"
    assert all_wrappers.wrappers[1].locator.resource_uri == "com.test"
    assert isinstance(all_wrappers.wrappers[0].locator, DummyLocator)
    assert isinstance(all_wrappers.wrappers[1].locator, DummyLocator)
    assert isinstance(all_wrappers.locator, StatelessLocator)
    assert all_wrappers.locator.resource_uri == "com.test"


def test_saveload():
    try:
        storage_db = {}
        input = {"in": 1}
        a = TestWrapper(offset=1, premap_key="in")
        b = TestWrapper(offset=2, premap_key="in")
        wrapper_list = [a, b]
        all_wrappers = WrapperPipeline(
            wrappers=wrapper_list, locator=DummyLocator(storage_db)
        )

        uid = all_wrappers.save_leaf()
        assert len(storage_db) == 3

        loaded_wrappers = WrapperPipeline(locator=DummyLocator(storage_db))
        loaded_wrappers = loaded_wrappers.load_leaf(uid)

        for i in range(2):
            assert all_wrappers.wrappers[i].offset == loaded_wrappers.wrappers[i].offset
            assert (
                all_wrappers.wrappers[i].premap_key
                == loaded_wrappers.wrappers[i].premap_key
            )
    finally:
        file_dir = str(pathlib.Path(__file__).parent.resolve())
        created_hash = "1921722fc520984398166966f5e4fad458c3e411"
        file_path = os.path.join(file_dir, created_hash)
        if os.path.exists(file_path):
            shutil.rmtree(file_path)


def test_saveload_linear():
    try:
        file_dir = str(pathlib.Path(__file__).parent.resolve())
        file_path = os.path.join(file_dir, "tmp")
        os.mkdir(file_path)

        input = {"in": 1}
        a = TestWrapper(offset=1, premap_key="in")
        b = SaveWrapper()
        wrapper_list = [a, b]
        all_wrappers = WrapperPipeline(wrappers=wrapper_list, resource_uri=file_path)

        all_wrappers.locator = FileLocator(file_path)
        # in real usage, you would use LinearLocators for all wrappers
        all_wrappers.wrappers[0].locator = FileLocator(file_path)

        uid = all_wrappers.save_leaf()
        assert len(os.listdir(file_path)) == 3

        loaded_wrappers = WrapperPipeline(locator=FileLocator(file_path))
        loaded_wrappers = loaded_wrappers.load_leaf(uid)

        for i in range(1):
            assert all_wrappers.wrappers[i].offset == loaded_wrappers.wrappers[i].offset
            assert (
                all_wrappers.wrappers[i].premap_key
                == loaded_wrappers.wrappers[i].premap_key
            )

    finally:
        if os.path.exists(file_path):
            shutil.rmtree(file_path)
