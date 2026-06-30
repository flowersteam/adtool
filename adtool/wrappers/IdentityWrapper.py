from typing import Dict, List

from adtool.wrappers.TransformWrapper import TransformWrapper
from adtool.utils.leaf.locators.locators import BlobLocator


class IdentityWrapper(TransformWrapper):
    """
    LEGACY: kept for compatibility with existing explorer/map defaults while
    the generic wrapper abstraction is phased out.

    Wrapper which passes the input without mutation.
    """

    def __init__(self, premap_keys: List[str] = []) -> None:
        super().__init__()
        # here only for explicitness
        # as the parent class already does it
        self.locator = BlobLocator()

    def map(self, input: Dict) -> Dict:
        return super().map(input)
