import json
import pickle
from typing import Any, Callable, Dict, Type

import numpy as np
import torch
from adtool.utils.leaf.Leaf import Leaf


class _JSONEncoderFactory:
    def __call__(self, dir_path: str, 
                 custom_callback: Callable
                 ):
        # return _CustomJSONENcoder but with a class attr dir_path
        cls = _CustomJSONEncoder
        cls._dir_path = dir_path
        cls._custom_callback = custom_callback

        return cls


class _CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        # catch torch Tensors
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        # catch numpy arrays
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # catch bytes
        # if isinstance(obj, bytes):
        #     return self._custom_callback(obj, self._dir_path)
        # catch Leaf objects
        if isinstance(obj, Leaf):
            uid = obj.save_leaf(self._dir_path)
            return str(uid)
        # catch python objects not serializable by JSON
        # this is only to comply with legacy code, as others should
        # implement Leaf
        try:
            json.JSONEncoder.default(self, obj)
        except TypeError:
        #    return  
    #      raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
            bin = pickle.dumps(obj)
            #current key
            return self._custom_callback(bin, self._dir_path,
                                         "pickle"
                                         )

        # pass to usual encoder
        return json.JSONEncoder.default(self, obj)


class SaveDiscovery:
    def __init__(self, **kwargs) -> None:
        """
        Dummy init which accepts any arguments, for backwards compatibility.
        """
        pass

    def __call__(
        self,
        resource_uri: str,
        experiment_id: int,
        run_idx: int,
        seed: int,
        discovery: Dict[str, Any],
    ) -> None:
        dir_path = self._initialize_save_path(
            resource_uri, experiment_id, run_idx, seed
        )

     #   raise NotImplementedError

        # create JSON encoder
        json_encoder = _JSONEncoderFactory()(
            dir_path=dir_path, 
             custom_callback=self._save_binary_callback
        )


        # save dict_data
        self._dump_json(
            discovery=discovery,
            dir_path=dir_path,
            json_encoder=json_encoder,
            experiment_id=experiment_id,
            run_idx=run_idx,
            seed=seed,
        )

        return

    @staticmethod
    def _dump_json(
        discovery: Dict[str, Any],
        dir_path: str,
        json_encoder: Type[json.JSONEncoder],
        **kwargs
    ) -> None:
        """
        Method called which dumps a human-readable JSON file.
        """
        raise NotImplementedError

    @staticmethod
    def _initialize_save_path(
        resource_uri: str, experiment_id: int, run_idx: int, seed: int
    ) -> str:
        """
        Formats discovery save path under the resource_uri,
        creating it if necessary
        """
        raise NotImplementedError

    @classmethod
    def _save_binary_callback(cls: Type, binary: bytes, save_dir: str) -> str:
        """
        Callback to call on binary objects when saving discovery. Its return
        value will be the value of the key in the serialized JSON object.
        """
        raise NotImplementedError
