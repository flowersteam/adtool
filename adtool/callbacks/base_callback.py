class BaseCallback:
    """
    Base class for callbacks used by the experiment pipelines when progress is made (e.g. new dicovery, explorer's optimization).
    """

    def __init__(self, logger=None, **kwargs) -> None:
        """
        initialize attributes common to all adtool.legacy callbacks

        Args:
            logger: logger used to report experiment progress.
        """
        self.logger = logger

    def __call__(self, experiment_id: int, seed: int, **kwargs) -> None:
        """
        The function to call to effectively raise the callback

        Args:
            experiment_id: current experiment id
            seed: current seed number
            kwargs: somme usefull parameters
        """
        raise NotImplementedError
