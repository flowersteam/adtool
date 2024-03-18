from adtool.callbacks import BaseCallback


class BaseOnErrorCallback(BaseCallback):
    """
    Base class for on error callbacks used by the experiment pipelines when an error are raise during the experiment progress.
    """

    def __init__(self, **kwargs) -> None:
        """
        initialize attributes common to all on error callbacks
        """
        super().__init__(**kwargs)

    def __call__(self, experiment_id: int, seed: int, **kwargs) -> None:
        """
        The function to call to effectively raise on error callback.
        Inform the user that the experiment are on error
        Args:
            experiment_id: current experiment id
            seed: current seed number
            kwargs: somme usefull parameters
        """
        print("Error for experiment {} with seed {}".format(experiment_id, seed))
