from adtool.callbacks import BaseCallback


class BaseOnSaveCallback(BaseCallback):
    """
    Base class for on save callbacks used by the experiment pipelines when the experiment save autodisc modules.
    """

    def __init__(self, **kwargs) -> None:
        """
        initialize attributes common to all on save callbacks
        """
        super().__init__(**kwargs)

    def __call__(self, experiment_id: int, seed: int, **kwargs) -> None:
        """
        The function to call to effectively raise on save callback.
        Inform the user that the experiment save are made
        Args:
            experiment_id: current experiment id
            seed: current seed number
            kwargs: somme usefull parameters
        """
        print("Saving experiment {} with seed {}".format(experiment_id, seed))
