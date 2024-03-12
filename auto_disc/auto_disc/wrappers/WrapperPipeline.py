from copy import deepcopy
from typing import Any, Dict, List, Union

from auto_disc.utils.leaf.Leaf import Leaf, Locator, StatelessLocator


class WrapperPipeline(Leaf):
    """
    Module for composing various wrappers during the input or output processing
    of the experiment.
    Usage example:
        ```
            input = {"in" : 1}
            a = alpha_wrapper()
            b = beta_wrapper()
            wrapper_list = [a, b]
            all_wrappers = WrapperPipeline(wrappers=wrapper_list,
                            inputs_to_save=["in"], outputs_to_save=["out"])
            assert all_wrappers.map(input) == b.map(a.map(input))
        ```
    """

    def __init__(
        self,
        wrappers: List["Leaf"] = [],
        resource_uri: str = "",
        locator: "Locator" = StatelessLocator(),
    ):
        super().__init__()

        # ensure mutual exclusivity of resource_uri and locator kwargs
        if (resource_uri != "") and (not isinstance(locator, StatelessLocator)):
            raise ValueError("Do not provide both `resource_uri` and `locator`.")
        elif resource_uri != "":
            self.locator.resource_uri = resource_uri
        elif not isinstance(locator, StatelessLocator):
            self.locator = locator
        else:
            pass

        # bind wrappers as submodules
        # NOTE: wrappers are therefore not individually
        # accessible by public methods
        for i, el in enumerate(wrappers):
            # do not need _set_attr_override as dicts are mutable
            self._modules[i] = el
            el.name = str(i)
            self._bind_wrapper_to_self(el)

        # # makes the dicts point to the same object, as dicts are mutable
        # self.wrappers = self._modules

    def __getattr__(self, name: str) -> Union[Any, "Leaf"]:
        if name == "wrappers":
            return self._modules
        else:
            return super().__getattr__(name)

    def _bind_submodule_to_self(self, submodule_name: str, submodule: "Leaf") -> None:
        raise Exception(
            """
                        Forbidden to bind submodules to WrapperPipeline
                        outside of initialization.
                        """
        )

    def _bind_wrapper_to_self(self, wrapper: "Leaf") -> None:
        # store pointer to parent container
        wrapper._set_attr_override("_container_ptr", self)

        # default initialization of locator resource_uri
        if isinstance(wrapper.locator, StatelessLocator):
            wrapper.locator = self.locator
        elif isinstance(wrapper.locator, Locator):
            # if Locator of wrappers are initialized, pass only resource_uri
            wrapper.locator.resource_uri = self.locator.resource_uri
        else:
            raise Exception("Locator uninitialized.")

        return

    def map(self, input: Dict) -> Dict:
        working_input = deepcopy(input)
        pipeline_length = len(self._modules)
        for i in range(pipeline_length):
            intermed_output = self._modules[i].map(working_input)
            working_input = intermed_output
        return working_input
