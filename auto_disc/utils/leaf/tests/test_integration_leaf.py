import os
from hashlib import sha1
from shutil import rmtree

from auto_disc.utils.leaf.Leaf import *
from auto_disc.utils.leaf.locators.Locator import *


class DiskPipeline(Leaf):
    def __init__(self):
        super().__init__()
        self.l1 = [1, 2, 3, 4]

    def create_locator(self, bin: bytes) -> "Locator":
        return DiskLocator(bin)

    def store_locator(self, loc):
        with open(f"/tmp/PipelineDir/{self.uid}", "wb") as f:
            f.write(loc.serialize())
        return

    @classmethod
    def retrieve_locator(cls, leaf_uid: str) -> "Locator":
        with open(f"/tmp/PipelineDir/{leaf_uid}", "rb") as f:
            loc = Locator.deserialize(f.read())
        return loc


class DiskLocator(Locator):
    def __init__(self, bin):
        self.uid = sha1(bin).hexdigest()

    def store(self, bin: bytes, *args, **kwargs) -> None:
        with open(f"/tmp/ResDir/{self.uid}", "wb") as f:
            f.write(bin)
        return

    def retrieve(self) -> bytes:
        with open(f"/tmp/ResDir/{self.uid}", "rb") as f:
            bin = f.read()
        return bin


def setup_function(function):
    global a
    a = DiskPipeline()
    os.mkdir("/tmp/PipelineDir")
    os.mkdir("/tmp/ResDir")
    return


# def test_diskpipeline_save_data():
#     # TODO: fix this test suite
#     a.save_leaf()
#     uid_old = a.uid

#     b = Leaf.load_leaf(uid_old)
#     assert b.l1 == [1, 2, 3, 4]


# def test_no_import_save_load_leaf():
#     # idk how to do it with conda, just manually run the no_import scripts and check
#     pass


def teardown_function(function):
    rmtree("/tmp/PipelineDir")
    rmtree("/tmp/ResDir")
    return


if __name__ == "__main__":
    setup_function(None)
    a.save_leaf()
    uid = a.uid
