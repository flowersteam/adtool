from ..shared import AnalysisModule
from .config import (
    load_space_coverage_config,
)
from .runner import run_space_coverage


class SpaceCoverageModule(AnalysisModule):
    module_id = "space_coverage"

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        self.module_config = load_space_coverage_config(self.config)

    def run(self, datasets, labels, run_dir) -> dict:
        return run_space_coverage(self.module_config, datasets, labels, run_dir)
