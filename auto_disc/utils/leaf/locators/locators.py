import os
from hashlib import sha1
from typing import Tuple

from auto_disc.utils.leaf.LeafUID import LeafUID
from auto_disc.utils.leaf.locators.LinearBase import FileLinearLocator
from auto_disc.utils.leaf.locators.Locator import FileLocator, Locator
from auto_disc.utils.leafutils.leafintegrations.expedb_locators import (
    ExpeDBLinearLocator,
    ExpeDBLocator,
)


class BlobLocator(Locator):
    """
    Default Locator class, for stateful modules.

    In reality, it is a factory which mimicks the interface of a Locator,
    and therefore lazy-initializes an appropriate Locator based on provided
    keywords.
    """

    def __init__(self, resource_uri: str = "", *args, **kwargs):
        self.resource_uri = resource_uri

    def store(self, bin: bytes, *args, **kwargs) -> "LeafUID":
        cls_to_init = self._parse_uri(self.resource_uri)
        locator = cls_to_init(resource_uri=self.resource_uri)

        return locator.store(bin, *args, **kwargs)

    def retrieve(self, uid: "LeafUID", *args, **kwargs) -> bytes:
        cls_to_init = self._parse_uri(self.resource_uri)
        locator = cls_to_init(resource_uri=self.resource_uri)

        return locator.retrieve(uid, *args, **kwargs)

    @classmethod
    def _parse_uri(cls, resource_uri: str) -> type:
        split_uri = resource_uri.split("//")
        if split_uri[0] == "http:":
            return ExpeDBLocator
        else:
            return FileLocator


class LinearLocator(Locator):
    """
    Bespoke Locator class, for stateful modules that need to store linear,
    potentially branching data.

    Also a factory.
    """

    def __init__(self, resource_uri: str = "", *args, **kwargs):
        self.resource_uri = resource_uri
        self.parent_id = -1

    def store(self, bin: bytes, parent_id: int = -1, *args, **kwargs) -> "LeafUID":
        # default setting if not set at function call
        if (self.parent_id != -1) and (parent_id == -1):
            parent_id = self.parent_id

        cls_to_init = self._parse_uri(self.resource_uri)
        locator = cls_to_init(resource_uri=self.resource_uri)

        # lazy initialized locator is temporary, so pass the parent_id to it
        # at runtime
        uid = locator.store(bin=bin, parent_id=parent_id, *args, **kwargs)

        # update parent_id after storage
        self.parent_id = locator.parent_id

        return uid

    def retrieve(self, uid: "LeafUID", length: int = 1, *args, **kwargs) -> bytes:
        cls_to_init = self._parse_uri(self.resource_uri)
        locator = cls_to_init(resource_uri=self.resource_uri)

        bin = locator.retrieve(uid, length)

        # update parent_id after retrieval
        self.parent_id = locator.parent_id

        return bin

    @classmethod
    def _parse_uri(cls, resource_uri: str) -> type:
        split_uri = resource_uri.split("//")
        if split_uri[0] == "http:":
            return ExpeDBLinearLocator
        else:
            return FileLinearLocator

    @classmethod
    def parse_bin(cls, bin: bytes) -> Tuple[str, bytes]:
        return FileLinearLocator.parse_bin(bin)

    @classmethod
    def parse_leaf_uid(cls, uid: LeafUID) -> Tuple[str, int]:
        db_name, node_id = uid.split(":")
        return FileLinearLocator.parse_leaf_uid(LeafUID(db_name), node_id)

    @staticmethod
    def hash(bin: bytes) -> "LeafUID":
        return FileLinearLocator.hash(bin)
