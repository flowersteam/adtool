import typing

from adtool.auto_disc.utils.callbacks import BaseCallback
from torch import Tensor


class BaseExpeDBCallback(BaseCallback):
    """
    Base class for callbacks used by the expe_db when progress is made (e.g. new dicovery, explorer's optimization).
    """

    def __init__(self, base_url: str, **kwargs) -> None:
        """
        initialize attributes common to all expe_db callbacks

        Args:
            base_url: the database url
            kwargs: some usefull args (e.g. logger experiment_id...)
        """
        super().__init__(**kwargs)
        self.base_url = base_url

    def _serialize_autodisc_space(self, space) -> typing.Dict:
        """
        Serialize an autodisc_space to make it savable as pickle or json

        Args:
            space: the autodisc space we want save as pickle or json
        Returns:
            serialized_space: A dict version savable as pickle or json
        """
        serialized_space = {}
        for key in space:
            if isinstance(space[key], Tensor):
                serialized_space[key] = space[key].tolist()
            else:
                serialized_space[key] = space[key]
        return serialized_space
