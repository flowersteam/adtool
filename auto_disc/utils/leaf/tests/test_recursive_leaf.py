import os
import pickle
import shutil

from auto_disc.utils.leaf.Leaf import Leaf, LeafUID
from auto_disc.utils.leaf.locators.Locator import FileLocator, Locator
from auto_disc.utils.leaf.tests.test_leaf import DummyLocator, DummyModule

"""
TODO: This test suite is pretty broken, need to revisit it when documenting
the new Leaf API. See #188, #216
"""


class DummyStatefulModule(Leaf):
    def __init__(self, s=None):
        super().__init__()
        self.locator = FileLocator()
        self.internal_state = s

    def forward(self, x):
        return [x + y for y in self.internal_state]


class DummyPipeline(Leaf):
    def __init__(self, l1=[1, 2, 3, 4], l2=[5, 6, 7, 8], resource_uri=""):
        super().__init__()
        self.locator = DummyLocator(resource_uri)
        self.l1 = DummyModule(l1)
        self.l2 = DummyModule(l2)


class DummyContained(Leaf):
    def __init__(self, s=None):
        super().__init__()
        self.internal_state = s

    def retrieve_metadata(self):
        return self._container_state["metadata"]


class DummyContainer(Leaf):
    def __init__(self, resource_uri):
        super().__init__()
        self.locator = DummyLocator(resource_uri)
        self.l1 = DummyContained([1, 2, 3, 4])
        self.l2 = DummyContained([1, 2, 3, 4])
        self.metadata = 42


class DummyFileLocator(Locator):
    def __init__(self, resource_uri: str = "", data_filename: str = "data"):
        # set default to relative directory of the caller
        if resource_uri == "":
            self.resource_uri = str(os.getcwd())
        else:
            self.resource_uri = resource_uri
        self.data_filename = data_filename

    def store(self, bin: bytes, *args, **kwargs) -> "LeafUID":
        uid = self.hash(bin)
        save_dir = os.path.join(self.resource_uri, str(uid))
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        data_save_path = os.path.join(save_dir, self.data_filename)
        with open(data_save_path, "wb") as f:
            f.write(bin)

        # save metadata only
        metadata_save_path = os.path.join(save_dir, "metadata")
        loaded_obj = DummyModule().deserialize(bin)
        loaded_obj.internal_state = []
        md_bin = loaded_obj.serialize()
        with open(metadata_save_path, "wb") as f:
            f.write(md_bin)

        return uid

    def retrieve(self, uid: "LeafUID", *args, **kwargs) -> bytes:
        save_dir = os.path.join(self.resource_uri, str(uid))

        # retrieve only the data saved, not the metadata
        data_save_path = os.path.join(save_dir, self.data_filename)
        with open(data_save_path, "rb") as f:
            bin = f.read()

        return bin


def setup_function(function):
    global a, res_uri
    func_name_arr = function.__name__.split("_")
    if "pipeline" in func_name_arr:
        global PipelineDB
        PipelineDB = {}
        res_uri = PipelineDB
        a = DummyPipeline(resource_uri=res_uri)
    elif "container" in func_name_arr:
        global ContainerDB
        ContainerDB = {}
        res_uri = ContainerDB
        a = DummyContainer(res_uri)
    elif "mixed" in func_name_arr:
        res_uri = os.path.join(os.getcwd(), "tmp")
        os.mkdir(res_uri)

        a = DummyPipeline(resource_uri=res_uri)

        # override the in-memory locator for file storage
        a.locator = FileLocator(resource_uri=res_uri)
        a.l1.locator = DummyFileLocator(resource_uri=res_uri, data_filename="data")
        a.l2.locator = DummyFileLocator(resource_uri=res_uri, data_filename="data")
    return


def teardown_function(function):
    func_name_arr = function.__name__.split("_")
    if ("mixed" in func_name_arr) and (os.path.exists(res_uri)):
        shutil.rmtree(res_uri)


def test_pipeline___init__():
    assert isinstance(a, Leaf)
    assert isinstance(a.l1, Leaf)
    assert isinstance(a.l2, Leaf)
    assert a.l1 == a._modules["l1"]
    assert a.l2 == a._modules["l2"]
    assert a.locator.resource_uri is PipelineDB

    assert a.l1.locator.resource_uri is PipelineDB
    assert a.l2.locator.resource_uri is PipelineDB


def test_pipeline_mutate_data():
    a.l1.internal_state = a.l1.forward(1)
    assert a.l1.internal_state == [2, 3, 4, 5]
    assert a.l2.internal_state == [5, 6, 7, 8]


def test_pipeline_container_ptr():
    assert a.l1._container_ptr is a
    assert a.l2._container_ptr is a


def test_pipeline_container_subleaf_names():
    assert a.l1.name == "l1"
    assert a.l2.name == "l2"


def test_pipeline_serialize_recursively():
    bin = a.serialize()
    obj = pickle.loads(bin)
    assert obj._modules["l1"] == a.l1._get_uid_base_case()
    assert obj._modules["l2"] == a.l2._get_uid_base_case()


def test_pipeline_save_data():
    uid_old = a.save_leaf()
    assert len(res_uri) == 3

    a.l1.internal_state = a.l1.forward(1)
    uid_new = a.save_leaf()
    # only one of the submodules is modified
    # assert len(res_uri) == 5

    b = DummyPipeline(l1=[], l2=[])
    b.locator = DummyLocator(res_uri)
    b = b.load_leaf(uid_old, res_uri)
    assert b.l1.internal_state == [1, 2, 3, 4]
    assert b.l1._container_ptr == b
    assert b.l2._container_ptr == b

    # after loading, it cannot remember to use the same locator as parent
    assert b.l2.locator != b.l1.locator != b.locator

    c = DummyPipeline(l1=[], l2=[])
    c.locator = DummyLocator(res_uri)
    c = c.load_leaf(uid_new, res_uri)
    assert c.l1.internal_state == [2, 3, 4, 5]
    assert c.l1._container_ptr == c
    assert c.l2._container_ptr == c


def test_pipeline_submodule_pointers():
    a.l3 = a.l2
    assert a.l2 == a.l3


def test_mixed___init__():
    assert isinstance(a.l1.locator, DummyFileLocator)
    assert isinstance(a.l2.locator, DummyFileLocator)


def test_mixed_save_leaf():
    uid = a.save_leaf()

    assert len(os.listdir(res_uri)) == 3

    data_ctr = 0
    metadata_ctr = 0
    for _, _, files in os.walk(res_uri):
        if "data" in files:
            data_ctr += 1
        if "metadata" in files:
            metadata_ctr += 1
    assert metadata_ctr == 3
    assert data_ctr == 2

    sample_save_path = os.path.join(
        res_uri, "8d633c07d386681ff8e1637d38934e7a8938a2b9" + "/metadata"
    )
    with open(sample_save_path, "rb") as f:
        bin = f.read()
    loaded_obj = pickle.loads(bin)
    assert loaded_obj.locator.split(".")[-1] == "DummyFileLocator"
    assert len(loaded_obj.internal_state) == 0


def test_mixed_load_leaf():
    uid = a.save_leaf()

    b = DummyPipeline()
    b.locator = FileLocator(resource_uri=res_uri)
    loaded_obj = b.load_leaf(uid, resource_uri=res_uri)

    assert loaded_obj.l1.internal_state == [1, 2, 3, 4]
    assert loaded_obj.l2.internal_state == [5, 6, 7, 8]


def test_container___init__():
    assert a.l1.locator.resource_uri == a.locator.resource_uri
    assert a.l2.locator.resource_uri == a.locator.resource_uri


def test_container_call_inner():
    assert a.l1.retrieve_metadata() == 42
    a.metadata = 16
    assert a.l1.retrieve_metadata() == 16


def test_container_serialize_recursively():
    bin = a.serialize()
    obj = pickle.loads(bin)
    assert isinstance(obj._modules["l1"], str)
    assert isinstance(obj._modules["l2"], str)
    assert obj._modules["l1"] == a.l1._get_uid_base_case()
    assert obj._modules["l2"] == a.l2._get_uid_base_case()
    assert obj._modules["l1"] != obj._modules["l2"]
