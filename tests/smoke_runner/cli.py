"""Command-line interface for smoke-test execution."""

from __future__ import annotations

import argparse
import sys

from .execution import run_single_test
from .specs import filter_specs, load_specs


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Run adtool smoke tests")
    parser.add_argument(
        "--only",
        nargs="*",
        default=[],
        help="Run only selected smoke tests by name (e.g. --only grayscott)",
    )
    parser.add_argument(
        "--refresh-baseline",
        action="store_true",
        help="Regenerate baseline ranges from current results",
    )
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Keep per-test runtime artifacts under tests/.tmp",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed test metrics",
    )
    args = parser.parse_args()

    specs = load_specs()
    if not specs:
        print("No smoke tests found under tests/*/test_spec.json")
        return 1

    try:
        selected_specs = filter_specs(specs, args.only)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    all_passed = True
    for spec in selected_specs:
        if not spec.enabled:
            reason = spec.skip_reason or "disabled in test_spec.json"
            print(f"[SKIP] {spec.name} - {reason}")
            continue

        print(f"[RUN] {spec.name}")
        passed, details = run_single_test(
            spec,
            refresh_baseline_flag=args.refresh_baseline,
            keep_artifacts=args.keep_artifacts,
            verbose=args.verbose,
        )

        if passed:
            print(f"[PASS] {spec.name}")
            continue

        all_passed = False
        print(f"[FAIL] {spec.name}")
        for detail in details:
            print(f"  - {detail}")

    return 0 if all_passed else 1
