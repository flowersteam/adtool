from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from adtool.utils.factory import instantiate_object


@dataclass(frozen=True)
class AnalysisModuleSpec:
    path: str
    config: dict = field(default_factory=dict)


class AnalysisModule(ABC):
    module_id = "analysis_module"

    def __init__(self, **config) -> None:
        self.config = dict(config)

    @property
    def identifier(self) -> str:
        return str(self.module_id)

    @abstractmethod
    def run(self, datasets, labels, run_dir) -> dict:
        pass


def load_analysis_module(spec: AnalysisModuleSpec) -> AnalysisModule:
    return instantiate_object(spec, object_name="analysis module")
