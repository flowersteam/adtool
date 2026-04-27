#!/usr/bin/env python3
"""Smoke-test runner entrypoint.

Implementation is split under tests/smoke_runner for maintainability.
"""

from __future__ import annotations

from tests.smoke_runner.cli import main as run_cli


def main() -> int:
    """Run the smoke-test CLI."""
    return run_cli()


if __name__ == "__main__":
    raise SystemExit(main())
