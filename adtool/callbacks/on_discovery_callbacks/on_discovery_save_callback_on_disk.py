import os
import pickle

from adtool.callbacks.on_discovery_callbacks import (
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
        comment: callback saves the discoveries outputs we want to save on disk.
        always saved : run_idx(pickle), experiment_id(pickle), seed(pickle)
        saved if key is in self.to_save_outputs: raw_run_parameters(pickle)
        run_parameters,(pickle)
        raw_output(pickle),
        output(pickle),
        rendered_outputs(changes according to the render function of the current system)
        Args:
            kwargs: run_idx, experiment_id, seed...
        """
        for save_item in self.to_save_outputs:
            folder = f"{self.folder_path}{kwargs['experiment_id']}/{kwargs['seed']}/{save_item}"
            
            if save_item != "rendered_outputs":
                filename = f"{folder}/idx_{kwargs['run_idx']}.pickle"
                if not os.path.isdir(folder):
                    os.makedirs(folder)
                with open(filename, "wb") as out_file:
                    pickle.dump(kwargs[save_item], out_file)
            else:
                for idx, (file_content, extension) in enumerate(kwargs["rendered_outputs"]):
                    subfolder = f"{folder}/output_{idx}"
                    if not os.path.isdir(subfolder):
                        os.makedirs(subfolder)
                    filename = f"{subfolder}/idx_{kwargs['run_idx']}.{extension}"
                    with open(filename, "wb") as out_file:
                        out_file.write(file_content.getbuffer())
            
        print(
            f"Saved in '{self.folder_path}' discovery {kwargs['run_idx']} for experiment {kwargs['experiment_id']}"
        )
        
        folder = f"{self.folder_path}{kwargs['experiment_id']}/{kwargs['seed']}/"
        self.logger.info(
            f"New discovery saved : {folder} : {self.to_save_outputs} :{kwargs['run_idx']}"
        )