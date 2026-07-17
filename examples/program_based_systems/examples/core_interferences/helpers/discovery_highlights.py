from __future__ import annotations

from typing import Any

import numpy as np

from adtool.user_tools.visu.highlights import (
    DiscoveryHighlightField,
    DiscoveryHighlightProvider,
    DiscoveryHighlightRule,
)


class CoreInterferenceDiscoveryHighlights(DiscoveryHighlightProvider):
    def fields(self) -> list[DiscoveryHighlightField]:
        return [
            DiscoveryHighlightField(
                field_id="l2_miss_core0",
                label="Core0 L2 miss",
                value_type="number",
                min=0,
                max=512,
            ),
            DiscoveryHighlightField(
                field_id="l2_miss_core1",
                label="Core1 L2 miss",
                value_type="number",
                min=0,
                max=512,
            ),
            DiscoveryHighlightField(
                field_id="l2_hit_core0",
                label="Core0 L2 hit",
                value_type="number",
                min=0,
                max=512,
            ),
            DiscoveryHighlightField(
                field_id="l2_hit_core1",
                label="Core1 L2 hit",
                value_type="number",
                min=0,
                max=512,
            ),
            DiscoveryHighlightField(
                field_id="diff_time",
                label="Mutual timing gap",
                value_type="number",
                min=-512,
                max=512,
            ),
            DiscoveryHighlightField(
                field_id="core0_program_lines",
                label="Core0 program lines",
                value_type="number",
                min=0,
                max=256,
            ),
            DiscoveryHighlightField(
                field_id="core1_program_lines",
                label="Core1 program lines",
                value_type="number",
                min=0,
                max=256,
            ),
        ]

    def rules(self) -> list[DiscoveryHighlightRule]:
        return [
            DiscoveryHighlightRule(
                rule_id="core0_l2_miss_range",
                label="Core0 L2 miss",
                field_id="l2_miss_core0",
                clauses=[{
                    "lower": 10,
                    "upper": 512,
                }],
            ),
            DiscoveryHighlightRule(
                rule_id="core1_l2_miss_range",
                label="Core1 L2 miss",
                field_id="l2_miss_core1",
                clauses=[{
                    "lower": 10,
                    "upper": 512,
                }],
            ),
            DiscoveryHighlightRule(
                rule_id="core0_program_lines_range",
                label="Core0 program lines",
                field_id="core0_program_lines",
                clauses=[{
                    "lower": 40,
                    "upper": 256,
                }],
            ),
            DiscoveryHighlightRule(
                rule_id="core1_program_lines_range",
                label="Core1 program lines",
                field_id="core1_program_lines",
                clauses=[{
                    "lower": 40,
                    "upper": 256,
                }],
            ),
        ]

    def compute_filters(self, discovery_payload: dict[str, Any]) -> dict[str, Any]:
        mutual = discovery_payload["raw_output"]["mutual"]
        dynamic_params = discovery_payload["params"]["dynamic_params"]

        return {
            "l2_miss_core0": _scalar(mutual["L2_miss_core0"]),
            "l2_miss_core1": _scalar(mutual["L2_miss_core1"]),
            "l2_hit_core0": _scalar(mutual["L2_hit_core0"]),
            "l2_hit_core1": _scalar(mutual["L2_hit_core1"]),
            "diff_time": _scalar(mutual["diff_time"]),
            "core0_program_lines": len(dynamic_params["core0"]),
            "core1_program_lines": len(dynamic_params["core1"]),
        }


def _scalar(value: Any) -> float:
    return float(np.asarray(value, dtype=float).reshape(-1)[0])
