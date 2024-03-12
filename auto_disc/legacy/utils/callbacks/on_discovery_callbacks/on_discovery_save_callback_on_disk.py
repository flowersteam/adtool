import os
import pickle

from auto_disc.legacy.utils.callbacks.on_discovery_callbacks import (
    BaseOnDiscoveryCallback,
)


class OnDiscoverySaveCallbackOnDisk(BaseOnDiscoveryCallback):
    """
    class for save experiment discovery on disk.
    """

    def __init__(self, folder_path, to_save_outputs, **kwargs) -> None:
        """
        init the callback with a path to save discoveries on disk

        Args:
            folder_path: path to folder where we want save the discovery
            to_save_outputs: list, key of "SAVABLE_OUTPUTS" (parent's attribute) to select the outputs who we want to save
        """
        super().__init__(to_save_outputs, **kwargs)
        self.folder_path = folder_path

    def __call__(self, **kwargs) -> None:
        """
        Save output when new discovery is made

        comment:callback save the discoveries outputs we want to save on disk.
                always saved : run_idx(pickle), experiment_id(pickle), seed(pickle)
                saved if key is in self.to_save_outputs: raw_run_parameters(pickle)
                                                    run_parameters,(pickle)
                                                    raw_output(pickle),
                                                    output(pickle),
                                                    rendered_output(changes according to the render function of the current system)
        Args:
            kwargs: run_idx, experiment_id, seed...
        """
        for save_item in self.to_save_outputs:
            folder = "{}{}/{}/{}".format(
                self.folder_path, kwargs["experiment_id"], kwargs["seed"], save_item
            )
            if save_item != "rendered_output":
                filename = "{}/idx_{}.pickle".format(folder, kwargs["run_idx"])
            else:
                filename = "{}/idx_{}.{}".format(
                    folder, kwargs["run_idx"], kwargs["rendered_output"][1]
                )

            if not os.path.isdir(folder):
                os.makedirs(folder)
            with open(filename, "wb") as out_file:
                if save_item != "rendered_output":
                    pickle.dump(kwargs[save_item], out_file)
                else:
                    out_file.write(kwargs["rendered_output"][0].getbuffer())
        print(
            "Saved in '{}' discovery {} for experiment {}".format(
                self.folder_path, kwargs["run_idx"], kwargs["experiment_id"]
            )
        )
        folder = "{}{}/{}/".format(
            self.folder_path, kwargs["experiment_id"], kwargs["seed"]
        )
        self.logger.info(
            "New discovery saved : {} : {} :{}".format(
                folder, self.to_save_outputs, kwargs["run_idx"]
            )
        )
