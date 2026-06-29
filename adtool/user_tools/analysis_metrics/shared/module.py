from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .imports import load_dotted_object


@dataclass(frozen=True)
class AnalysisModuleSpec:
    path: str
    config: dict = field(default_factory=dict)


class AnalysisModule(ABC):
    module_id = "analysis_module"

    def __init__(self, config: dict | None = None) -> None:
        self.config = dict(config or {})

    @property
    def identifier(self) -> str:
        return str(self.module_id)

    @abstractmethod
    def run(self, datasets, labels, run_dir) -> dict:
        pass


def load_analysis_module(spec: AnalysisModuleSpec) -> AnalysisModule:
    module_cls = load_dotted_object(spec.path)
    return module_cls(spec.config)
