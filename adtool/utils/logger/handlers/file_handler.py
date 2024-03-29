import logging
from logging import FileHandler


class SetFileHandler(FileHandler):
    """
    Handler to put logs in file on disk
    """

    def __init__(self, folder_log_path: str, experiment_id: int) -> None:
        """
        Init an handler able to save log in file on disk

        Args:
            folder_log_path: The path to the folder where the user want save logs
            experiment_id: current experiment id
        """
        FileHandler.__init__(
            self, "{}exp_{}.log".format(folder_log_path, experiment_id)
        )
        self.setLevel(logging.NOTSET)
        self.setFormatter(
            logging.Formatter(
                "%(name)s - %(levelname)s - SEED %(seed)s - LOG_ID %(id)s - %(message)s"
            )
        )
        self.experiment_id = experiment_id
