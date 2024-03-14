import os
from copy import deepcopy
from typing import Dict, Iterable, List

from adtool.wrappers.TransformWrapper import TransformWrapper
from adtool.utils.leaf.Leaf import Leaf, LeafUID, Locator
from adtool.utils.leaf.locators.LinearBase import (
    Stepper,
    retrieve_packed_trajectory,
    retrieve_trajectory,
)
from adtool.utils.leaf.locators.locators import LinearLocator


class SaveWrapper(TransformWrapper):
    """
    Wrapper which does basic processing and
    saving of captured *input* history
    Usage example:
        ```
            input = {"in" : 1}
            # default setting saves all specified `premap_keys`
            wrapper = SaveWrapper(premap_keys = ["in"],
                                  postmap_keys = ["out"])
            output = wrapper.map(input)
            assert output["out"] == 1
        ```
    """

    def __init__(
        self,
        premap_keys: List[str] = [],
        postmap_keys: List[str] = [],
        inputs_to_save: List[str] = [],
    ) -> None:
        super().__init__(premap_keys=premap_keys, postmap_keys=postmap_keys)

        # resource_uri should be defined when SaveWrapper is bound to a
        # container, or manually initialized
        self.locator = LinearLocator("")

        # MUST define with the name `buffer` to obey LinearStorage interface
        self.buffer = []

        # save all inputs by default
        if len(inputs_to_save) == 0:
            self.inputs_to_save = premap_keys
        else:
            self.inputs_to_save = inputs_to_save

    def map(self, input: Dict) -> Dict:
        """
        WARN: This wrapper's .map() is stateful.

        Transforms the input dictionary with the provided relabelling of keys,
        saving inputs and passing outputs
        """
        # must do because dicts are mutable types
        intermed_dict = deepcopy(input)

        if len(self.inputs_to_save) > 0:
            self._store_saved_inputs_in_buffer(intermed_dict)
        else:
            self._store_all_inputs_in_buffer(intermed_dict)

        output = self._transform_keys(intermed_dict)

        return output

    def save_leaf(self, resource_uri: str, parent_uid: int = -1) -> "LeafUID":
        # parent_uid is passed for specifying the parent node,
        # when passed to LinearLocator.store() by super().save_leaf()
        uid = super().save_leaf(resource_uri, parent_uid)

        # clear cache
        self.buffer = []
        return uid

    def serialize(self) -> bytes:
        """
        Custom serialize method needed for producing appropriately padded
        binary with metadata.
        """
        data_bin = super().serialize()

        # store buffer
        old_buffer = self.buffer
        del self.buffer

        # create metadata hash which defines the name of the SQLite db
        metadata_bytehash = bytes.fromhex(self.locator.hash(super().serialize()))

        # pad to create output_bin
        output_bin = metadata_bytehash + bytes.fromhex("deadbeef") + data_bin

        # restore buffer
        self.buffer = old_buffer

        return output_bin

    def get_history(self, lookback_length: int = 1) -> List[Dict]:
        """
        Retrieve the history of inputs to the wrapper.
        `lookback_length = -1` corresponds to retrieving the entire history,
        which may be very large as it will query the on-disk SQLite db.
        """
        if lookback_length == 1:
            history_buffer = deepcopy(self.buffer)
        else:
            # retrieve buffers from SQLite db
            # this check does not affect the lookback_length = -1 case
            if lookback_length > 1:
                retrieval_length = lookback_length - 1
            else:
                # it will only be -1 here or 0
                retrieval_length = lookback_length

            try:
                retrieved_buffer = self._retrieve_buffer(
                    self.locator.resource_uri, retrieval_length
                )
            except Exception as e:
                if (
                    "unable to open database file" in str(e)
                    and self.locator.resource_uri != ""
                ):
                    # in this branch we assume that if the resource_uri is
                    # set and the db is not found, then the db has not been
                    # created yet, and the entire history is in memory,
                    # so we return an empty buffer here
                    retrieved_buffer = []
                else:
                    # otherwise, we raise an exception suggesting that
                    # the resource_uri is not set
                    raise e

            # attach to working buffer in memory
            copy_buffer = deepcopy(self.buffer)
            retrieved_buffer.extend(copy_buffer)

            history_buffer = retrieved_buffer

        return history_buffer

    def _retrieve_buffer(self, buffer_src_uri: str, length: int) -> List[Dict]:
        """
        Temporarily query SQLite db to retrieve buffer, taking the top-level
        resource_uri and the length of the buffer to retrieve.

        Does not store buffer as attribute of the SaveWrapper instance,
        but does load everything into memory.

        `length = -1` corresponds as usual to retrieving the entire history.
        """
        temporary_locator = LinearLocator(resource_uri=buffer_src_uri)

        # run part of the save routine to ge the db name
        # TODO: can tighten this up with direct calls
        tmp_bin = self.serialize()
        db_name, _ = temporary_locator.parse_bin(tmp_bin)

        # get parent_id of last insert for retrieval
        parent_id = self.locator.parent_id

        # assemble db_url
        db_url = os.path.join(buffer_src_uri, db_name, "lineardb")

        bin = retrieve_trajectory(db_url, parent_id, length)
        # run usual Locator retrieve routine
        stepper = Stepper().deserialize(bin)

        return stepper.buffer

    def generate_dataloader(
        self, buffer_src_uri: str, cachebuf_size: int = 1
    ) -> Iterable[Dict]:
        """
        Query SQLite DB to retrieve an iterable which lazy loads the DB tree.
        """
        return BufferStreamer(
            wrapper=self, resource_uri=buffer_src_uri, cachebuf_size=cachebuf_size
        )

    def _store_saved_inputs_in_buffer(self, intermed_dict: Dict) -> None:
        saved_input = {}
        for key in self.inputs_to_save:
            saved_input[key] = intermed_dict[key]
        self.buffer.append(saved_input)
        return

    def _store_all_inputs_in_buffer(self, intermed_dict: Dict) -> None:
        saved_input = deepcopy(intermed_dict)
        self.buffer.append(saved_input)
        return

    def _transform_keys(self, old_dict: Dict) -> Dict:
        return super()._transform_keys(old_dict)

    # def _get_uid_base_case(self) -> 'LeafUID':
    #     """
    #     Override pointerization so that the proper UID is stored
    #     """
    #     padded_bin = self.serialize()
    #     db_name, _ = self.locator._parse_bin(padded_bin)

    #     return db_name


class BufferStreamer:
    """
    Class for streaming buffers of arbitrary length from SQLite db.

    `mode` is set to one of "serial" or "batched" which specifies whether
    the cachebuf is returned as a single item or as a list of items.

    """

    def __init__(
        self,
        wrapper: SaveWrapper,
        resource_uri: str,
        cachebuf_size: int = 1,
        mode: str = "serial",
    ) -> None:
        self.cachebuf_size = cachebuf_size
        self.mode = mode

        # variable for tracking where we are in the DB
        # starting at the id of the last insert
        self._i = wrapper.locator.parent_id

        # store working buffer
        self.buffer = []

        # assemble db_url
        self.wrapper = wrapper
        self.resource_uri = resource_uri
        db_name = self._get_db_name()
        self.db_url = os.path.join(self.resource_uri, db_name, "lineardb")

    def __iter__(self):
        # returns self with __next__ method dynamically set by mode
        # TODO: currently only supports serial mode
        return self

    def __next__(self):
        # return next item in buffer
        # dynamically overridden by mode specification, but
        # will call _next_serial() by default (for testing)
        return self._next_serial()

    def _next_serial(self) -> Dict:
        # simply pop things from buffer
        # if buffer is empty, retrieve next cachebuf
        if len(self.buffer) == 0:
            # check if we are at the end of the DB
            # based on SQLite convention that first row is 1
            if self._i < 1:
                raise StopIteration
            else:
                self.buffer = self._next_cachebuf()
        return self.buffer.pop(-1)

    def _next_batched(self) -> List[Dict]:
        # return the entire cachebuf at a time
        # i.e., as batches
        # check if we are at the end of the DB
        # based on SQLite convention that first row is 1
        if self._i < 1:
            raise StopIteration
        else:
            return self._next_cachebuf()

    def _get_db_name(self) -> str:
        """
        Get db name (i.e., hash of metadata) from SaveWrapper
        """
        padded_bin = self.wrapper.serialize()
        db_name, _ = self.wrapper.locator.parse_bin(padded_bin)

        return db_name

    def _next_cachebuf(self) -> List[Dict]:
        """
        Retrieves next cachebuf. This is where StopIteration is raised.
        """
        # retrieve next cachebuf and one extra to set _i
        ids, packed_trajectory, _ = retrieve_packed_trajectory(
            self.db_url, self._i, self.cachebuf_size + 1
        )

        # set to the next id to be retrieved
        self._i = ids[0]

        # check for the final cachebuf, which may have a misalignment
        if len(ids) < self.cachebuf_size + 1:
            # set to 0 to signal that entire tree has been crawled
            # for next time the method is called
            self._i = 0
            traj_to_concat = packed_trajectory
        else:
            # if not at the final cachebuf,
            # remove first packed binary which contains the elements
            # from the start of the next cachebuf
            traj_to_concat = packed_trajectory[1:]

        # unpack binaries in the trajectory
        buffer_concat = []
        for binary in traj_to_concat:
            loaded_obj = Stepper().deserialize(binary)
            buffer_concat += loaded_obj.buffer

        return buffer_concat
