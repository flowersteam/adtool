import codecs
import json
import os
import pickle
import tempfile
from typing import Dict, List, Tuple

import adtool.utils.leaf.locators.LinearBase as LinearBase
import requests
from adtool.utils.leaf.Leaf import LeafUID
from adtool.utils.leaf.locators.LinearBase import FileLinearLocator
from adtool.utils.leaf.locators.Locator import Locator
import sys

def _initialize_checkpoint(entrypoint: str, dict: Dict = {}) -> str:
    """
    Inserts a Mongo document and retrieves the unique key generated
    """
    response = requests.post(entrypoint, json=dict).json()
    return response["ID"]


def _query_uid(resource_uri: str, uid: LeafUID) -> List[Dict]:
    uid_filter = {"uid": uid}
    uid_filter = _format_filter_from_dict(uid_filter)
    response = requests.get(resource_uri + "?filter=" + uid_filter).json()
    return response


def _format_filter_from_dict(filter_dict: Dict) -> str:
    """
    Creates appropriate filter str from parsing a dict
    """
    filters = []
    for key in filter_dict.keys():
        if filter_dict[key] == None:
            pass
        elif isinstance(filter_dict[key], list):
            filters.append(
                '{{"{}":{{"$in":[{}]}} }}'.format(key), ", ".join(filter_dict[key])
            )
        else:
            if isinstance(filter_dict[key], str):
                filters.append('{{"{}":"{}" }}'.format(key, filter_dict[key]))
            else:
                filters.append('{{"{}":{} }}'.format(key, filter_dict[key]))

    # join array elements of filters to string filter
    if len(filters) > 1:
        filter = '{"$and":['
        for f in filters:
            filter += f + ","
        filter = filter[:-1]
        filter += "]}"
    else:
        filter = filters[0]
    return filter


class ExpeDBLocator(Locator):
    """
    Locator which saves modules à la Git, to ExpeDB,
    with the entrypoint URL (i.e., DB) specified by resource_uri
    """

    def __init__(self, resource_uri: str = ""):
        # strip trailing /
        if len(resource_uri) > 0 and resource_uri[-1] == "/":
            resource_uri = resource_uri[:-1]
        self.resource_uri = resource_uri

    def store(self, bin: bytes, *args, **kwargs) -> "LeafUID":
        uid = self.hash(bin)
        # encode in base64 so MongoDB doesn't escape anything
        bin = codecs.encode(bin, encoding="base64")

        mongo_id = self._retrieve_mongo_id(uid)

        # update the uid field
        # TODO: refactor me
        entrypoint_url = self.resource_uri + "/" + mongo_id
        doc_update_dict = {"uid": uid}
        requests.post(entrypoint_url, json=doc_update_dict)

        # attach files
        entrypoint_url = self.resource_uri + "/" + mongo_id + "/files"
        file_dict = {"metadata": bin}
        requests.post(entrypoint_url, files=file_dict)

        return uid

    def retrieve(self, uid: "LeafUID", *args, **kwargs) -> bytes:
        # extra info past : is for other locators
        uid = uid.split(":")[0]

        mongo_id = self._retrieve_mongo_id(uid)
        response_bin = requests.get(
            self.resource_uri + "/" + mongo_id + "/metadata"
        ).content
        bin = codecs.decode(response_bin, encoding="base64")
        return bin

    def _retrieve_mongo_id(self, uid: LeafUID) -> str:
        response = _query_uid(self.resource_uri, uid)
        print('_retrieve_mongo_id response', response)
        if len(response) == 0:
            mongo_id = _initialize_checkpoint(self.resource_uri)
        elif len(response) == 1:
            mongo_id = response[0]["_id"]
        else:
            raise Exception("ExpeDB is corrupted with duplicate checkpoints.")

        return mongo_id


class ExpeDBLinearLocator(Locator):
    """
    Locator which stores branching, linear data to ExpeDB,
    with the entrypoint URL (i.e., DB) specified by resource_uri
    """

    def __init__(self, resource_uri: str = ""):
        # strip trailing /
        if len(resource_uri) > 0 and resource_uri[-1] == "/":
            resource_uri = resource_uri[:-1]
        self.resource_uri = resource_uri
        self.parent_id = -1

    def store(self, bin: bytes, parent_id: int = -1) -> "LeafUID":
        # default setting if not set at function call
        if (self.parent_id != -1) and (parent_id == -1):
            parent_id = self.parent_id

        # init cachedir
        cache_dir = self._init_cache_dir()

        # parse formatted binary
        lineardb_name, data_bin = FileLinearLocator.parse_bin(bin)

        # store metadata binary
        mongo_id = self._retrieve_tree_and_store_metadata(
            cache_dir, lineardb_name, data_bin
        )

        # store data binary
        row_id = LinearBase.store_data(cache_dir, data_bin, parent_id)
        # update parent_id in instance
        self.parent_id = int(row_id)

        # return leaf_uid
        leaf_uid = lineardb_name + ":" + str(row_id)

        # dump sqlite to binary
        db_url = os.path.join(cache_dir, "lineardb")
        with open(db_url, "rb") as f:
            lineardb_dump = f.read()
        lineardb_dump = codecs.encode(lineardb_dump, encoding="base64")

        # update lineardb on remote
        entrypoint_url = self.resource_uri + "/" + mongo_id + "/files"
        file_dict = {"lineardb": lineardb_dump}
        requests.post(entrypoint_url, files=file_dict)

        # delete cachedir
        os.remove(os.path.join(cache_dir, "lineardb"))
        os.rmdir(cache_dir)

        return leaf_uid

    def retrieve(self, uid: "LeafUID", length: int = 1) -> bytes:
        try:
            lineardb_name, row_id = uid.split(":")
            # set parent_id from retrieval
            self.parent_id = int(row_id)
        # check in case too many strings are returned
        except ValueError:
            raise ValueError("leaf_uid is not properly formatted.")

        mongo_id = _query_uid(self.resource_uri, lineardb_name)[0]["_id"]
        cache_dir = self._init_cache_dir()
        self._load_tree_into_cache_dir(cache_dir, mongo_id)

        db_url = os.path.join(cache_dir, "lineardb")

        bin = LinearBase.retrieve_trajectory(
            db_url=db_url, row_id=row_id, length=length
        )

        return bin

    def _init_cache_dir(self) -> str:
        return tempfile.mkdtemp()

    def _retrieve_tree_and_store_metadata(
        self, cache_dir: str, lineardb_name: str, data_bin: bytes
    ) -> str:
        """
        Retrieves existing LinearDB from resource_uri, or creating if doesn't
        exist.
        Also stores metadata when creating the LinearDB is required.

        Returns the ObjectID from ExpeDB.
        """

        response = _query_uid(self.resource_uri, lineardb_name)
        if len(response) == 0:
            # first-save logic below
            # make new chkpt and associate the uid
            mongo_id = _initialize_checkpoint(
                self.resource_uri, dict={"uid": lineardb_name}
            )

            # parse metadata
            loaded_obj = pickle.loads(data_bin)
            del loaded_obj.buffer
            metadata_bin = pickle.dumps(loaded_obj)

            # save metadata
            entrypoint_url = self.resource_uri + "/" + mongo_id + "/files"
            file_dict = {"metadata": metadata_bin}
            requests.post(entrypoint_url, files=file_dict)

        elif len(response) == 1:
            mongo_id = response[0]["_id"]
            # no need to resave metadata
            self._load_tree_into_cache_dir(cache_dir, mongo_id)
        else:
            raise Exception("ExpeDB is corrupted with duplicate checkpoints.")

        return mongo_id

    def _create_tree(cache_dir: str) -> None:
        """
        Create SQLite cache.
        """
        db_path = os.path.join(cache_dir, "lineardb")
        LinearBase.init_db(db_path)
        return

    def _load_tree_into_cache_dir(self, cache_dir: str, mongo_id: str) -> None:
        print(self.resource_uri + "/" + mongo_id + "/lineardb",file=sys.stderr)
        response_bin = requests.get(
            self.resource_uri + "/" + mongo_id + "/lineardb"
        )
        if response_bin.status_code != 200:
            raise Exception("Failed to retrieve lineardb from ExpeDB.")
        
        bin = codecs.decode(response_bin.content, encoding="base64")
        lineardb_path = os.path.join(cache_dir, "lineardb")
        with open(lineardb_path, "wb") as f:
            f.write(bin)
        return
