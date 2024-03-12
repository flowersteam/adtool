import codecs
import json
import os

import auto_disc.utils.leafutils.leafintegrations.expedb_locators as expedb_locators
import requests
from auto_disc.auto_disc.wrappers.SaveWrapper import SaveWrapper
from auto_disc.utils.leaf.locators.LinearBase import FileLinearLocator, Stepper
from auto_disc.utils.leafutils.leafintegrations.expedb_locators import (
    ExpeDBLinearLocator,
    ExpeDBLocator,
    _format_filter_from_dict,
    _initialize_checkpoint,
)

import sys

def setup_function(function):
    global RESOURCE_URI
    RESOURCE_URI = "http://127.0.0.1:5001/checkpoint_saves"




def teardown_function(function):
    pass


def test_ExpeDBLocator_init():
    loc = ExpeDBLocator()


def test_ExpeDBLocator__format_filter_from_dict():
    filter_dict = {"uid": "abcd", "metadata": 2}
    filter_str = _format_filter_from_dict(filter_dict)
    assert filter_str == '{"$and":[{"uid":"abcd" },{"metadata":2 }]}'


def test_ExpeDBLocator__initialize_checkpoint():
    loc = ExpeDBLocator(resource_uri=RESOURCE_URI)
    id = _initialize_checkpoint(RESOURCE_URI)

    # small test for the 24-character hexadecimal string
    assert len(id) == 24


def test_ExpeDBLocator__retrieve_mongo_id():
    def mock_checkpoint():
        from uuid import uuid1

        entrypoint = RESOURCE_URI
        uid = str(uuid1())
        response = json.loads(
            requests.post(entrypoint, json={"uid": uid, "metadata": "123"}).content
        )
        return response["ID"], uid

    mongo_id, uid = mock_checkpoint()
    loc = ExpeDBLocator(resource_uri=RESOURCE_URI)
    retrieved_id = loc._retrieve_mongo_id(uid=uid)
    assert retrieved_id == mongo_id


def test_ExpeDBLocator_store():
    bin = b"123"
    loc = ExpeDBLocator(resource_uri=RESOURCE_URI)
    uid = loc.store(bin)
    print('uid', uid)

    # test by making a manual API call
    mongo_id = loc._retrieve_mongo_id(uid)
    print('mongo_id', mongo_id)
    response_bin = requests.get(RESOURCE_URI + "/" + mongo_id + "/metadata").content

    print('response_bin', response_bin)

    response_bin = codecs.decode(response_bin, encoding="base64")

    

    assert response_bin == bin

    # TODO: test that duplicate inserts result in update


def test_ExpeDBLocator_retrieve():
    bin = b"a23[9tg[s9guusehf[v08g[8g\]]]]"
    loc = ExpeDBLocator(resource_uri=RESOURCE_URI)
    uid = loc.store(bin)

    retrieved_bin = loc.retrieve(uid)

    print("retrieved_bin", retrieved_bin)

    assert bin == retrieved_bin


def test_ExpeDBLinearLocator___init__():
    loc = ExpeDBLinearLocator()


def test_ExpeDBLinearLocator__init_cache_dir():
    loc = ExpeDBLinearLocator()
    filepath = loc._init_cache_dir()
    assert isinstance(filepath, str)
    assert os.path.exists(filepath)
    os.rmdir(filepath)





def test_ExpeDBLinearLocator_store():
    saver = SaveWrapper()
    saver.buffer = [bytes(1), bytes(2)]
    bin = saver.serialize()
    dbname, _ = FileLinearLocator.parse_bin(bin)

    print("dbname", dbname,file=sys.stderr)

    # delete doc if exists
    response_arr = expedb_locators._query_uid(RESOURCE_URI, dbname)
    if len(response_arr)==1:
        mongo_id = response_arr[0]["_id"]
        entrypoint_url = os.path.join(RESOURCE_URI, mongo_id)
        requests.delete(entrypoint_url)
    elif len(response_arr)>1:
        raise Exception("Weird, there are multiple documents with the same name")



    # store
    loc = ExpeDBLinearLocator(resource_uri=RESOURCE_URI)
    uid = loc.store(bin, parent_id=-1)
    uid = loc.store(bin, parent_id=1)
    uid = loc.store(bin, parent_id=2)

    assert uid.split(":")[-1] == "3"


def test_ExpeDBLinearLocator__retrieve_tree_and_store_metadata():
    def no_response(a, b):
        return []

    # test initialize path
 #  expedb_locators._query_uid = no_response
    loc = ExpeDBLinearLocator(resource_uri=RESOURCE_URI)
    cache_dir = loc._init_cache_dir()
    stepper = Stepper()
    stepper.buffer = [bytes(1), bytes(2)]
    data_bin = stepper.serialize()
    

    mongo_id=loc._retrieve_tree_and_store_metadata(cache_dir, "lineardb", data_bin)

    print("mongo_id", mongo_id)



def test_ExpeDBLinearLocator_retrieve():
    saver = SaveWrapper()
    saver.buffer = [bytes(1), bytes(2)]
    bin = saver.serialize()

    loc = ExpeDBLinearLocator(resource_uri=RESOURCE_URI)
    uid = loc.store(bin, parent_id=-1)
    uid = loc.store(bin, parent_id=1)

    bin = loc.retrieve(uid, length=2)
    stepper = Stepper().deserialize(bin)
    assert stepper.buffer == [bytes(1), bytes(2), bytes(1), bytes(2)]
