import os
import pickle

from adtool.auto_disc.utils.callbacks.interact_callbacks import BaseInteractCallback


class SaveDiskInteractCallback(BaseInteractCallback):
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

    def __call__(self, data, dict_info=None, **kwargs) -> None:
        """
        The function to call to effectively raise cancelled callback.
        Inform the user that the experience is in canceled status
        Args:
            experiment_id: current experiment id
            seed: current seed number
            kwargs: somme usefull parameters
        """
        kwargs["data"] = data
        kwargs["dict_info"] = dict_info

        item_to_save = ["data", "dict_info"]
        for save_item in item_to_save:
            folder = "{}/{}/{}/{}".format(
                self.folder_path,
                dict_info["experiment_id"],
                dict_info["seed"],
                save_item,
            )
            filename = "{}/idx_{}.pickle".format(folder, dict_info["idx"])

            if not os.path.isdir(folder):
                os.makedirs(folder)
            with open(filename, "wb") as out_file:
                pickle.dump(kwargs[save_item], out_file)
        folder = "{}{}/{}/".format(
            self.folder_path, dict_info["experiment_id"], dict_info["seed"]
        )
        self.logger.info(
            "New data saved : {} : {} :{}".format(
                folder, item_to_save, dict_info["idx"]
            )
        )
