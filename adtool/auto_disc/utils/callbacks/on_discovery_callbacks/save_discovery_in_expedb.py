import json
import os
import pickle
from datetime import datetime
from hashlib import sha1
from typing import Any, Dict, Type
from uuid import uuid1

import requests
import torch
from adtool.auto_disc.utils.callbacks.on_discovery_callbacks.save_discovery import (
    SaveDiscovery,
)


class SaveDiscoveryInExpeDB(SaveDiscovery):
    def __call__(
        self,
        resource_uri: str,
        experiment_id: int,
        run_idx: int,
        seed: int,
        discovery: Dict[str, Any],
    ) -> None:
        super().__call__(resource_uri, experiment_id, run_idx, seed, discovery)

        return

    @staticmethod
    def _dump_json(
        discovery: Dict[str, Any],
        dir_path: str,
        json_encoder: Type[json.JSONEncoder],
        **kwargs
    ) -> None:
        # converts discovery to JSON blob and calls the _save_binary_callback
        # when needed. Reloads the JSON blob as a dict to push to the DB
        json_blob = json.dumps(discovery, cls=json_encoder)

        # push to DB
        # NOTE: that the Python stdlib json encoder treats NaN/float(+-inf)
        # as strings, which is technically not valid JSON. This will cause an
        # error using the python requests library if one loads it in with the
        # json kwarg. The "solution" is to use the data kwarg instead, which
        # passes a dumb binary, and manually override the mimetype
        # Note, that this is "noncompliant" with JSON standards,
        # but it is JavaScript compliant,
        # see this PR: https://github.com/psf/requests/issues/5767
        #
        # Broken code example:
        # ```
        #     parsed_dict_data = json.loads(json_blob)
        #     requests.post(dir_path, json=parsed_dict_data)
        # ````
        response = requests.post(
            dir_path, data=json_blob, headers={"Content-Type": "application/json"}
        )
        print(response.text)

        return

    @staticmethod
    def _initialize_save_path(
        resource_uri: str, experiment_id: int, run_idx: int, seed: int
    ) -> str:
        """
        Pushes metadata to the MongoDB discoveries collection,
        and returns the ID of the newly created document.
        """
        # initial payload
        payload = {"experiment_id": experiment_id, "run_idx": run_idx, "seed": seed}
        response = requests.post(resource_uri + "/discoveries", json=payload)
        doc_id = json.loads(response.text)["ID"]

        document_path = os.path.join(resource_uri, "discoveries", doc_id)
        return document_path

    @classmethod
    def _save_binary_callback(cls: Type, binary: bytes, document_path: str) -> str:
        """
        Pushes binary data to the MongoDB discoveries collection in the form of
        top-level key-value pairs {sha1_hash : binary_data}
        """
        sha1_hash = sha1(binary).hexdigest()
        files_to_save = {sha1_hash: binary}

        requests.post(document_path + "/files", files=files_to_save)
        return sha1_hash
