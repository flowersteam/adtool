from .config import (
    load_comparison_2d_config,
)
from .runner import run_comparison_2d
from ..shared import AnalysisModule


class Comparison2DModule(AnalysisModule):
    module_id = "comparison_2d"

    def __init__(self, **config) -> None:
        super().__init__(**config)
        self.module_config = load_comparison_2d_config(self.config)

    def run(self, datasets, labels, run_dir) -> dict:
        return run_comparison_2d(self.module_config, datasets, labels, run_dir)
