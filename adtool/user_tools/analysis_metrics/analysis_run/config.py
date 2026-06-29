import json
from dataclasses import dataclass, field
from pathlib import Path

from ..shared import AnalysisModuleSpec


@dataclass(frozen=True)
class AnalysisRunConfig:
    analysis_modules: list[AnalysisModuleSpec] = field(default_factory=list)


def load_analysis_run_config(config_path):
    if config_path is None:
        return AnalysisRunConfig()

    with Path(config_path).open("r") as handle:
        payload = json.load(handle)

    raw_modules = payload.get("analysis_modules") or []
    return AnalysisRunConfig(
        analysis_modules=[
            AnalysisModuleSpec(
                path=module["path"],
                config=dict(module.get("config") or {}),
            )
            for module in raw_modules
        ],
    )
