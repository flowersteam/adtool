import json
import os
from datetime import datetime
from uuid import uuid1

from adtool.utils.callbacks.on_save_finished_callbacks import (
    BaseOnSaveFinishedCallback,
)
from adtool.utils.leaf.Leaf import LeafUID


class GenerateReport:
    """
    Takes UID which identifies a saved object and annotates the saved data with
    human-readable identifying information
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def __call__(
        self,
        report_dir: str = "",
        uid: LeafUID = LeafUID(""),
        experiment_id: int = 0,
        seed: int = 0,
        run_idx: int = 0,
        **kwargs,
    ) -> None:
        # construct filename
        dt = datetime.now()
        date_str = dt.isoformat(timespec="minutes")
        filename = f"{date_str}_exp_{experiment_id}_idx_{run_idx}_{uid}"
        file_path = os.path.join(report_dir, f"{filename}.json")
        report = {
            "uid": uid,
            "experiment_id": experiment_id,
            "seed": seed,
            "run_idx": run_idx,
            "metadata": kwargs.get("metadata", None),
        }

        with open(file_path, mode="w") as f:
            json.dump(report, f)

        # img_path = os.path.join(report_dir, f"{filename}_rendered_output")
        # with open(img_path, mode="wa") as f:
        #     f.write(rendered_output)

        return
