from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DiscoverySet:
    path: Path
    files: list
    payloads: list
    outputs: object


@dataclass(frozen=True)
class DatasetInfo:
    path: Path
    label: str
    count: int

    def to_payload(self) -> dict:
        return {
            "path": str(self.path),
            "label": self.label,
            "count": self.count,
        }


@dataclass(frozen=True)
class AnalysisImage:
    file: str
    title: str
    plot_type: str
    dimensions: list = field(default_factory=list)
    bounds: list = field(default_factory=list)

    def to_payload(self) -> dict:
        return {
            "file": self.file,
            "title": self.title,
            "plot_type": self.plot_type,
            "dimensions": list(self.dimensions),
            "bounds": [list(bounds) for bounds in self.bounds],
        }


@dataclass(frozen=True)
class AnalysisRunSummary:
    run_dir: Path
    dataset_a: DatasetInfo
    dataset_b: DatasetInfo
    module_order: list
    modules: dict

    def to_payload(self) -> dict:
        return {
            "run_dir": str(self.run_dir),
            "dataset_a": self.dataset_a.to_payload(),
            "dataset_b": self.dataset_b.to_payload(),
            "module_order": list(self.module_order),
            "modules": dict(self.modules),
        }
