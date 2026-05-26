"""Constants shared by the smoke-test runner modules."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TESTS_ROOT = ROOT / "tests"
TMP_ROOT = TESTS_ROOT / ".tmp"
DEFAULT_OUTPUT_KEY = "output"
DEFAULT_DYNAMIC_PARAMS_PATH = "params.dynamic_params"
