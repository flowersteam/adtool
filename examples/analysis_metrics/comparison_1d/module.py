from adtool.examples.analysis_metrics.comparison_1d.config import (
    load_comparison_1d_config,
)
from adtool.examples.analysis_metrics.comparison_1d.runner import run_comparison_1d
from adtool.examples.analysis_metrics.shared import AnalysisModule


class Comparison1DModule(AnalysisModule):
    module_id = "comparison_1d"

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        self.module_config = load_comparison_1d_config(self.config)

    def run(self, datasets, labels, run_dir) -> dict:
        return run_comparison_1d(self.module_config, datasets, labels, run_dir)
