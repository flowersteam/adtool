from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class DiscoveryHighlightField:
    field_id: str
    label: str
    value_type: str
    min: float | None = None
    max: float | None = None
    choices: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "field_id": self.field_id,
            "label": self.label,
            "value_type": self.value_type,
        }
        if self.min is not None:
            payload["min"] = self.min
        if self.max is not None:
            payload["max"] = self.max
        if self.choices is not None:
            payload["choices"] = self.choices
        return payload


@dataclass
class DiscoveryHighlightRule:
    rule_id: str
    label: str
    field_id: str
    clauses: list[dict[str, Any]]
    enabled_by_default: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "rule_id": self.rule_id,
            "label": self.label,
            "field_id": self.field_id,
            "clauses": self.clauses,
            "enabled_by_default": self.enabled_by_default,
        }
        return payload


class DiscoveryHighlightProvider(ABC):
    def __init__(self, **config: Any) -> None:
        self.config = config

    @abstractmethod
    def fields(self) -> list[DiscoveryHighlightField]:
        raise NotImplementedError

    @abstractmethod
    def rules(self) -> list[DiscoveryHighlightRule]:
        raise NotImplementedError

    @abstractmethod
    def compute_filters(self, discovery_payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def schema(self) -> dict[str, Any]:
        return {
            "fields": [field.to_dict() for field in self.fields()],
            "rules": [rule.to_dict() for rule in self.rules()],
        }
