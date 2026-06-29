from adtool.examples.analysis_metrics.comparison_2d.config import (
    load_comparison_2d_config,
)
from adtool.examples.analysis_metrics.comparison_2d.runner import run_comparison_2d
from adtool.examples.analysis_metrics.shared import AnalysisModule


class Comparison2DModule(AnalysisModule):
    module_id = "comparison_2d"

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        self.module_config = load_comparison_2d_config(self.config)

    def run(self, datasets, labels, run_dir) -> dict:
        return run_comparison_2d(self.module_config, datasets, labels, run_dir)
