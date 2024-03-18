from adtool.callbacks import BaseCallback


class CustomPrintCallback(BaseCallback):
    def __init__(self, custom_message_to_print: str, **kwargs) -> None:
        """
        Init the callback with a message to print
        Args:
            custom_message_to_print: The message to print
        """
        super().__init__(**kwargs)
        self._custom_message_to_print = custom_message_to_print

    def __call__(self, experiment_id: int, seed: int, **kwargs) -> None:
        """
        Print the message with contextutal information

        Args:
            experiment_id: current experiment id
            seed: current seed number
            kwargs: some usefull paramters (e.g. run_idx)
        """
        print(
            self._custom_message_to_print + " / Iteration: {}".format(kwargs["run_idx"])
        )
