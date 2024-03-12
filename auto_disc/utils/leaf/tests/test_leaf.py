import pickle
from hashlib import sha1

from auto_disc.utils.leaf.Leaf import Leaf
from auto_disc.utils.leaf.locators.Locator import Locator, StatelessLocator


class DummyModule(Leaf):
    def __init__(self, s=None):
        super().__init__()
        self.internal_state = s

    def forward(self, x):
        return [x + y for y in self.internal_state]


class DummyLocator(Locator):
    def __init__(self, resource_uri: dict = {}):
        self.resource_uri = resource_uri

    def store(self, bin):
        uid = self.hash(bin)
        self.resource_uri[uid] = bin
        return uid

    def retrieve(self, uid):
        return self.resource_uri[uid]


def setup_function(function):
    global a, ResDB, res_uri
    a = DummyModule([1, 2, 3, 4])
    ResDB = {}
    res_uri = ResDB
    return


def test_leaf_init():
    assert a._modules == {}
    assert isinstance(a.locator, StatelessLocator)
    assert a.name == ""
    assert a.internal_state


def test_leaf_serialize():
    bin = a.serialize()
    b = pickle.loads(bin)

    del a.locator
    del b.locator
    assert a.__dict__ == b.__dict__


def test_leaf_deserialize():
    bin = a.serialize()
    b = a.deserialize(bin)
    assert isinstance(a.locator, StatelessLocator)
    assert b.locator == "leaf.locators.Locator.StatelessLocator"
    del a.locator
    del b.locator
    assert a.__dict__ == b.__dict__


def test_locator_init():
    a.locator = DummyLocator(res_uri)
    assert isinstance(a.locator, Locator)
    assert a.locator.resource_uri == ResDB


def test_locator_store_retrieve():
    bin1 = a.serialize()
    a.locator = DummyLocator(res_uri)
    uid = a.locator.store(bin1)

    bin2 = a.locator.retrieve(uid)
    assert bin1 == bin2


def test_leaf_save_load():
    a.locator = DummyLocator(res_uri)
    uid = a.save_leaf()

    b = DummyModule()
    b.locator = DummyLocator(res_uri)
    # this loading overrides b.locator
    b = b.load_leaf(uid)
    assert b.internal_state == [1, 2, 3, 4]
    a.internal_state.append(5)
    a.internal_state = a.forward(1)

    # test uid updates after save_leaf
    new_uid = a.save_leaf()
    assert new_uid != uid

    b.locator = DummyLocator(res_uri)
    c = b.load_leaf(new_uid)
    assert c.internal_state == [2, 3, 4, 5, 6]
    assert b.internal_state == [1, 2, 3, 4]
    assert a.internal_state == c.internal_state != b.internal_state
