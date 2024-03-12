from typing import Dict, List

from auto_disc.auto_disc.wrappers.TransformWrapper import TransformWrapper
from auto_disc.utils.leaf.locators.locators import BlobLocator


class IdentityWrapper(TransformWrapper):
    """
    Wrapper which passes the input without mutation.
    """

    def __init__(self, premap_keys: List[str] = []) -> None:
        super().__init__()
        # here only for explicitness
        # as the parent class already does it
        self.locator = BlobLocator()

    def map(self, input: Dict) -> Dict:
        return super().map(input)
