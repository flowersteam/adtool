from __future__ import annotations

from typing import Any

import numpy as np


MATRIX_SUM_LABELS = [
    "miss_core0",
    "miss_core1",
    "hits_core0",
    "hits_core1",
    "miss_read_core0",
    "miss_read_core1",
    "miss_write_core0",
    "miss_write_core1",
    "hits_read_core0",
    "hits_read_core1",
    "hits_write_core0",
    "hits_write_core1",
]

SCALAR_LABELS = [
    "diff_time_core0",
    "diff_time_core1",
    "diff_time",
    "L2_miss_core0",
    "L2_miss_core1",
    "L2_hit_core0",
    "L2_hit_core1",
    "L2_miss_write_core0",
    "L2_miss_write_core1",
    "L2_miss_read_core0",
    "L2_miss_read_core1",
    "L2_hit_write_core0",
    "L2_hit_write_core1",
    "L2_hit_read_core0",
    "L2_hit_read_core1",
]

COMPACT_LABELS = MATRIX_SUM_LABELS + SCALAR_LABELS


def compact_interference_metrics(
    datasets: list[Any],
    config: dict,
) -> tuple[np.ndarray, ...]:
    _ = config
    return tuple(_compact(dataset.payloads) for dataset in datasets) + (COMPACT_LABELS,)


def _compact(payloads: list[dict[str, Any]]) -> np.ndarray:
    rows = [_compact_payload(payload) for payload in payloads]
    return np.asarray(rows, dtype=float)


def _compact_payload(payload: dict[str, Any]) -> list[float]:
    mutual = _mutual_metrics(payload)
    row = [_sum_metric(mutual, key) for key in MATRIX_SUM_LABELS]
    row.extend(_scalar_metric(mutual, key) for key in SCALAR_LABELS)
    return row


def _mutual_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    raw_output = payload.get("raw_output")
    if not isinstance(raw_output, dict):
        raise ValueError("Core interference pretreatment requires raw_output")

    mutual = raw_output.get("mutual")
    if not isinstance(mutual, dict):
        raise ValueError(
            "Core interference pretreatment requires raw_output['mutual']"
        )
    return mutual


def _sum_metric(mutual: dict[str, Any], key: str) -> float:
    return float(np.sum(_metric_array(mutual, key)))


def _scalar_metric(mutual: dict[str, Any], key: str) -> float:
    value = _metric_array(mutual, key)
    if value.size != 1:
        raise ValueError(f"Expected scalar metric for {key}, got {value.shape}")
    return float(value.reshape(-1)[0])


def _metric_array(mutual: dict[str, Any], key: str) -> np.ndarray:
    if key not in mutual:
        raise ValueError(f"Core interference raw metric is missing: {key}")

    value = np.asarray(mutual[key], dtype=float)
    if np.isnan(value).any() or np.isinf(value).any():
        raise ValueError(f"Core interference raw metric is not finite: {key}")
    return value
