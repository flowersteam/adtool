import os
import pathlib
import shutil
import tempfile
from copy import deepcopy
from typing import Dict, List

from auto_disc.auto_disc.wrappers import SaveWrapper, TransformWrapper, WrapperPipeline
from auto_disc.utils.leaf.Leaf import Leaf
from auto_disc.utils.leaf.LeafUID import LeafUID
from auto_disc.utils.leaf.locators.Locator import FileLocator, Locator, StatelessLocator


class IncrementerWrapper(Leaf):
    def __init__(self) -> None:
        super().__init__()

    def map(self, input: Dict) -> Dict:
        # must do because dicts are mutable types
        output = deepcopy(input)
        output["data"] = output["data"] + 1

        return output


class FakeExperimentPipeline(Leaf):
    def __init__(
        self,
        wrappers: List["Leaf"] = [],
        locator: "Locator" = StatelessLocator(),
        save_db_url: str = "",
    ) -> None:
        super().__init__()
        # pass save_db_url to wrappers
        self.input_wrappers = WrapperPipeline(wrappers, resource_uri=save_db_url)
        # pass locator to WrapperPipeline
        self.input_wrappers.locator = locator

        # pass locator to unset wrappers
        for wrapper in self.input_wrappers.wrappers.values():
            if isinstance(wrapper.locator, StatelessLocator):
                wrapper.locator = locator

        # use locator also for saving own metadata
        self.locator = locator

    def input_transformation(self, input: Dict) -> Dict:
        output = self.input_wrappers.map(input)
        return output


class DictLocator(Locator):
    def __init__(self, filepath):
        self.filepath = filepath

    def store(self, bin: bytes, *args, **kwargs) -> "LeafUID":
        uid = LeafUID(self.hash(bin))
        self.table[uid] = bin
        return uid

    def retrieve(self, uid: "LeafUID", *args, **kwargs) -> bytes:
        return self.table[uid]


def setup_function(function):
    global FILE_PATH
    file_dir = str(pathlib.Path(__file__).parent.resolve())
    FILE_PATH = os.path.join(file_dir, "tmp")
    os.mkdir(FILE_PATH)


def teardown_function():
    if os.path.exists(FILE_PATH):
        shutil.rmtree(FILE_PATH)


def test___init__():
    wrappers = [
        SaveWrapper(),
        IncrementerWrapper(),
        TransformWrapper(premap_keys=["data"], postmap_keys=["output"]),
        SaveWrapper(),
    ]
    pipeline = FakeExperimentPipeline(wrappers, save_db_url=FILE_PATH)
    assert isinstance(pipeline._modules["input_wrappers"], WrapperPipeline)
    assert pipeline.input_wrappers.locator.resource_uri == ""
    assert pipeline.input_wrappers.wrappers[0].locator.resource_uri == FILE_PATH
    pipeline = FakeExperimentPipeline(
        wrappers, save_db_url="test", locator=StatelessLocator(FILE_PATH)
    )
    assert pipeline.input_wrappers.locator.resource_uri == FILE_PATH
    assert pipeline.input_wrappers.wrappers[0].locator.resource_uri == "test"


def test_input_transformation():
    input = {"data": 1}
    wrappers = [
        SaveWrapper(),
        IncrementerWrapper(),
        TransformWrapper(premap_keys=["data"], postmap_keys=["output"]),
        SaveWrapper(),
    ]
    pipeline = FakeExperimentPipeline(wrappers)
    output = pipeline.input_transformation(input)
    assert output == {"output": 2}


def test_saveload():
    input = {"data": 1}
    wrappers = [
        TransformWrapper(premap_keys=["output"], postmap_keys=["data"]),
        SaveWrapper(),
        IncrementerWrapper(),
        TransformWrapper(premap_keys=["data"], postmap_keys=["output"]),
        SaveWrapper(),
    ]
    pipeline = FakeExperimentPipeline(
        wrappers, save_db_url=FILE_PATH, locator=FileLocator(FILE_PATH)
    )
    assert pipeline.locator.resource_uri == FILE_PATH
    assert pipeline.input_wrappers.locator.resource_uri == FILE_PATH

    output = pipeline.input_transformation(input)
    pipeline.input_transformation(output)

    pipeline_uid = pipeline.save_leaf()
    assert len(os.listdir(FILE_PATH)) == 7

    new_pipeline = FakeExperimentPipeline(
        save_db_url=FILE_PATH, locator=FileLocator(FILE_PATH)
    )
    new_pipeline = new_pipeline.load_leaf(pipeline_uid, FILE_PATH)
    assert new_pipeline.input_wrappers.wrappers[3].premap_keys == ["data"]
    assert new_pipeline.input_wrappers.wrappers[3].postmap_keys == ["output"]
    assert new_pipeline.input_wrappers.wrappers[1].buffer == [{"data": 1}, {"data": 2}]
    assert new_pipeline.input_wrappers.wrappers[4].buffer == [
        {"output": 2},
        {"output": 3},
    ]
