import os
import pathlib
import pickle
import shutil
from hashlib import sha1

from adtool.utils.leaf.Leaf import Leaf
from adtool.utils.leaf.locators.Locator import Locator
from adtool.utils.leaf.locators.locators import BlobLocator


class Node(Leaf):
    def __init__(self, data, ptr: "Node" = None):
        super().__init__()
        self.locator = BlobLocator()
        self.data = data
        self.ptr = ptr


class TestLinkedList:
    def setup_method(self, method):
        global RESOURCE_URI
        file_path = str(pathlib.Path(__file__).parent.resolve())
        RESOURCE_URI = os.path.join(file_path, "tmp")
        os.mkdir(RESOURCE_URI)
        return

    def teardown_method(self, method):
        global RESOURCE_URI
        if os.path.exists(RESOURCE_URI):
            shutil.rmtree(RESOURCE_URI)
        return

    def test_Node(self):
        save_uid = Node(1, Node(2, Node(3))).save_leaf(resource_uri=RESOURCE_URI)

        loaded = Node(0).load_leaf(save_uid, resource_uri=RESOURCE_URI)
        assert loaded.data == 1
        assert loaded.ptr.data == 2
        assert loaded.ptr.ptr.data == 3
