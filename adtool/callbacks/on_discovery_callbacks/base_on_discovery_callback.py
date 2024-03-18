from typing import List

from adtool.callbacks import BaseCallback


class BaseOnDiscoveryCallback(BaseCallback):
    """
    Base class for on discovery callbacks used by the experiment pipelines when new discovery is made.
    """

    SAVABLE_OUTPUTS = [
        "raw_run_parameters",
        "run_parameters",
        "raw_output",
        "output",
        "rendered_output",
    ]

    def __init__(self, to_save_outputs: List[str], **kwargs) -> None:
        """
        initialize attributes common to all on discovery callbacks

        Args:
            to_save_outputs: Names of all outputs the user want save
            kwargs: some usefull args (e.g. experiment_id...)
        """
        super().__init__(**kwargs)
        self.to_save_outputs = to_save_outputs

    def __call__(self, experiment_id: int, seed: int, **kwargs) -> None:
        """
        The function to call to effectively raise on discovery callback.
        Inform the user that the experiment as made new discovery
        Args:
            experiment_id: current experiment id
            seed: current seed number
            kwargs: somme usefull parameters
        """
        print(
            "New discovery for experiment {} with seed {}".format(experiment_id, seed)
        )
