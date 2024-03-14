from abc import ABCMeta, abstractmethod
from typing import Any, Dict

from adtool.utils.leaf.Leaf import Leaf
from adtool.utils.leaf.locators.locators import BlobLocator


class System(Leaf, metaclass=ABCMeta):
    """An abstract class that defines the interface for a `System`, and should
    be inherited from by concrete implementations of `System`s.

    A `System` is a model for a dynamical or complex system. In the context
    of the experimental pipeline, it should be stateless and therefore a pure
    function. Therefore, while the system parameters may need to be set,
    these are in fact _hyperparameters_ set by the web interface, and not
    parameters which are explored by the curiosity algorithm.

    As a pure function, it takes a payload of data as input and returns an
    new payload of data as output, without mutating the input.
    """

    @abstractmethod
    def __init__(self, *args, **kwargs):
        """
        There are no assumptions on the initialization parameters.
        """
        super().__init__()
        self.locator = BlobLocator()

    @abstractmethod
    def map(self, input: Dict) -> Dict:
        """Map system inputs to outputs.

        A `System` operates on regular Python dicts, but it views them as
        structured. The `premap_key` and `postmap_key` are used to define
        the structured elements that the `System` operates on. Often, the
        `postmap_key` does not exist in the input dict, and is added by the
        `Map` as output.

        Whether or not the `premap_key` exists in the output dict is up to the
        implementation of the specific `System`. We recommend preserving it.

        Args:
            input (dict):
                generic dict containing input data to the map at `premap_key`
        Returns:
            A generic dict containing output data from the map at `postmap_key`
        """
        raise NotImplementedError

    @abstractmethod
    def render(self, data_dict: dict) -> bytes:
        """Render system output as an image or video.

        A `System` should be able to render useful information about its
        execution (e.g., a plot, a video, etc.), which may depend on its
        internal state which is not captured by the `map` method.

        Args:
            self (System):
                If the `System` can render a graphical output without
                needing access to its internal state (i.e., only from the
                output), then it can also be `@classmethod`.
            data_dict (dict):
                A dict containing the output of the `map`
                method. Depending on implementation, this may be sufficient to
                render the output or one may need stateful data from `self`.
        """
        raise NotImplementedError
