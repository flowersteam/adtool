import os
import pickle
from time import sleep

from auto_disc.legacy.utils.callbacks.interact_callbacks import BaseInteractCallback


class ReadDiskInteractCallback(BaseInteractCallback):
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
        self.folder_path = kwargs["folder_path"]

    def __call__(self, filter_attribut=None, **kwargs) -> None:
        """
        The function to call to effectively raise cancelled callback.
        Inform the user that the experience is in canceled status
        Args:
            experiment_id: current experiment id
            seed: current seed number
            kwargs: somme usefull parameters
        """
        self.logger.info("start read data files")
        data_path = "{}{}/{}/{}/idx_{}.pickle".format(
            self.folder_path,
            filter_attribut["experiment_id"],
            filter_attribut["seed"],
            "data",
            filter_attribut["idx"],
        )
        dict_info_path = "{}{}/{}/{}/idx_{}.pickle".format(
            self.folder_path,
            filter_attribut["experiment_id"],
            filter_attribut["seed"],
            "dict_info",
            filter_attribut["idx"],
        )
        self.logger.info(data_path)
        self.logger.info(dict_info_path)
        timer = 1
        while not os.path.exists(data_path) or not os.path.exists(data_path):
            if timer < 60:
                timer *= 2
            sleep(timer)
        response = None
        while response == None or response["type"] != "answer":
            self.logger.info("try read data")
            if timer < 60:
                timer *= 2
            with open(dict_info_path, "rb") as dictFile:
                response = pickle.load(dictFile)
            with open(data_path, "rb") as dataFile:
                response["file_data"] = pickle.load(dataFile)
            self.logger.info("fin try read data")
            sleep(timer)
        self.logger.info("stop read data path: " + response["type"])
        return response
