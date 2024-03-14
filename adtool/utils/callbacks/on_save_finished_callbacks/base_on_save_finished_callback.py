from adtool.utils.callbacks import BaseCallback


class BaseOnSaveFinishedCallback(BaseCallback):
    """
    Base class for on save finished callbacks used by the experiment pipelines when the experiment save autodisc modules.
    """

    def __init__(self, **kwargs) -> None:
        """
        initialize attributes common to all on save finished callbacks
        """
        super().__init__(**kwargs)

    def __call__(self, experiment_id: int, seed: int, **kwargs) -> None:
        """
        The function to call to effectively raise on save finished callback.
        Inform the user that the experiment save are over
        Args:
            experiment_id: current experiment id
            seed: current seed number
            kwargs: somme usefull parameters
        """
        print("Experiment {} with seed {} saved".format(experiment_id, seed))
