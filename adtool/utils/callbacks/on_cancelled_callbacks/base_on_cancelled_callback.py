from adtool.utils.callbacks import BaseCallback


class BaseOnCancelledCallback(BaseCallback):
    """
    Base class for cancelled callbacks used by the experiment pipelines when the experiment was cancelled.
    """

    def __init__(self, **kwargs) -> None:
        """
        initialize attributes common to all cancelled callbacks

        Args:
            kwargs: some usefull args (e.g. experiment_id...)
        """
        super().__init__(**kwargs)

    def __call__(self, experiment_id: int, seed: int, **kwargs) -> None:
        """
        The function to call to effectively raise cancelled callback.
        Inform the user that the experience is in canceled status
        Args:
            experiment_id: current experiment id
            seed: current seed number
            kwargs: somme usefull parameters
        """
        print("Experiment {} with seed {} cancelled".format(experiment_id, seed))
