from typing import Dict, List

from adtool.wrappers.TransformWrapper import TransformWrapper


class SaveWrapper(TransformWrapper):
    """
    Compatibility shim for the removed legacy history wrapper.

    Explorer history is now handled exclusively by `adtool.explorers.history_store`.
    This wrapper only preserves the old key-relabelling interface for any
    remaining non-history call sites.
    """

    def __init__(
        self,
        premap_keys: List[str] = [],
        postmap_keys: List[str] = [],
        inputs_to_save: List[str] = [],
    ) -> None:
        super().__init__(premap_keys=premap_keys, postmap_keys=postmap_keys)
        self.inputs_to_save = list(inputs_to_save)

    def map(self, input: Dict) -> Dict:
        return super().map(input)
