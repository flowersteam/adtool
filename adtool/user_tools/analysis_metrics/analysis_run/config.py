import json
from dataclasses import dataclass, field
from pathlib import Path

from adtool.utils.factory import coerce_object_spec
from ..shared import AnalysisModuleSpec


@dataclass(frozen=True)
class AnalysisRunConfig:
    analysis_modules: list[AnalysisModuleSpec] = field(default_factory=list)
    checkpoint_name: str | None = None


def load_analysis_run_config(config_path):
    if config_path is None:
        return AnalysisRunConfig()

    with Path(config_path).open("r") as handle:
        payload = json.load(handle)

    raw_modules = payload.get("analysis_modules") or []
    checkpoint_name = payload.get("checkpoint_name")
    if checkpoint_name is not None and not isinstance(checkpoint_name, str):
        raise ValueError("analysis checkpoint_name must be a string or null")
    return AnalysisRunConfig(
        checkpoint_name=checkpoint_name,
        analysis_modules=[
            AnalysisModuleSpec(
                path=spec.path,
                config=spec.config,
            )
            for module in raw_modules
            for spec in [coerce_object_spec(module, object_name="analysis module")]
        ],
    )
