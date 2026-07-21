# Smoke Tests

This folder hosts smoke tests for each example in the project. Each test lives in its own subfolder and is fully configuration-driven.

## Folder layout

Each test folder must contain:
- smoke_config.json: lightweight experiment config used for smoke runs
- test_spec.json: execution metadata for the runner
- baseline_ranges.json: generated acceptance ranges (created by the runner)

Example:
- tests/my_example/
  - smoke_config.json
  - test_spec.json
  - baseline_ranges.json

## How to add a new test

1) Create a new folder under tests/
- Example: tests/my_example/

2) Add smoke_config.json
- Start from the example config in the corresponding example folder, then make it faster:
  - Reduce cycles, bootstrap_size, num_instructions, etc.
  - Keep only the required callbacks (usually SaveDiscoveryOnDisk).
  - Consider `"render_every": 0` when rendered media is not needed for the smoke metric.
- The runner will override save_location at runtime.

3) Add test_spec.json
- Use this template:

{
  "name": "my_example",
  "config_file": "smoke_config.json",
  "nb_iterations": 5,
  "seed": 42,
  "experiment_id": 9999,
  "output_key": "output",
  "dynamic_params_path": "params.dynamic_params"
}

Notes:
- output_key should match where the system writes output in discovery.json.
- dynamic_params_path should match the nested params location.

Optional fields:
- enabled: false
- skip_reason: "Explain why it should be skipped"
- compare_metrics: [ "discovery_count", "output_dim", ... ]

4) Generate baseline_ranges.json
- Run:
  python3 tests/run_tests.py --only my_example --refresh-baseline

This writes baseline_ranges.json to the test folder.

## Running tests

- Run all tests:
  python3 tests/run_tests.py

- Run a subset:
  python3 tests/run_tests.py --only my_example other_example

- Refresh baselines:
  python3 tests/run_tests.py --only my_example --refresh-baseline

## How the runner works

The runner logic lives under tests/smoke_runner and does the following:
- Loads test specs from tests/*/test_spec.json
- Runs `python -m adtool.runners.run_experimentations` with a temporary config and isolated save directory
- Reads discoveries from the run output
- Computes generic numeric metrics
- Compares metrics to baseline_ranges.json
- Supports per-test skip rules and optional compare_metrics filtering

## Tips for stable smoke tests

- Prefer small cycles/iterations and minimal bootstrap_size.
- Use compare_metrics when a domain is noisy or unstable.
- If output is highly stochastic, focus on robust metrics like:
  - discovery_count, output_dim, output_finite_ratio, seconds_per_iteration
- Always refresh baselines after intentional behavior changes.
