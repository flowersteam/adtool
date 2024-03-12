from copy import deepcopy
from typing import Dict, List

from auto_disc.utils.leaf.Leaf import Leaf
from auto_disc.utils.leaf.locators.locators import BlobLocator


class TransformWrapper(Leaf):
    """
    Wrapper which does basic processing of input
    Usage example:
        ```
            input = {"in" : 1}
            wrapper = TransformWrapper(premap_keys = ["in"],
                                  postmap_keys = ["out"])
            output = wrapper.map(input)
            assert output["out"] == 1
        ```
    """

    def __init__(
        self,
        premap_keys: List[str] = [],
        postmap_keys: List[str] = [],
    ) -> None:
        super().__init__()
        self.locator = BlobLocator()

        # process key wrapping
        if len(premap_keys) != len(postmap_keys):
            raise ValueError("premap_keys and transformed_keys must be same length.")
        else:
            pass

        self.premap_keys = premap_keys
        self.postmap_keys = postmap_keys

    def map(self, input: Dict) -> Dict:
        """
        Transforms the input dictionary with the provided relabelling of keys.
        """
        # must do because dicts are mutable types
        intermed_dict = deepcopy(input)

        output = self._transform_keys(intermed_dict)

        return output

    def _transform_keys(self, old_dict: Dict) -> Dict:
        # initialize empty dict so that key-values will not overwrite
        new_dict = {}
        for old_key, new_key in zip(self.premap_keys, self.postmap_keys):
            # allows making conditional transformers that ignore input
            # with no appropriately matching keys
            if old_dict.get(old_key, None) is not None:
                new_dict[new_key] = old_dict[old_key]
                del old_dict[old_key]

        new_dict.update(old_dict)

        return new_dict
