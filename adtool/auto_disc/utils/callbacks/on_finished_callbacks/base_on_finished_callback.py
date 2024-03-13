from adtool.auto_disc.utils.callbacks import BaseCallback


class BaseOnFinishedCallback(BaseCallback):
    """
    Base class for on finished callbacks used by the experiment pipelines when the experiment is over.
    """

    def __init__(self, **kwargs) -> None:
        """
        initialize attributes common to all on finished callbacks
        """
        super().__init__(**kwargs)

    def __call__(self, experiment_id: int, seed: int, **kwargs) -> None:
        """
        The function to call to effectively raise on finished callback.
        Inform the user that the experiment are over
        Args:
            experiment_id: current experiment id
            seed: current seed number
            kwargs: somme usefull parameters
        """
        print("Experiment {} with seed {} finished".format(experiment_id, seed))
