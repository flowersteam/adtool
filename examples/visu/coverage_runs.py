from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import HTTPException

if __package__:
    from .runtime import ServerConfig
    from .server_support import is_relative_to
else:
    from runtime import ServerConfig
    from server_support import is_relative_to


def coverage_runs_dir(config: ServerConfig) -> Path:
    return (config.discoveries.parent / "coverage_runs").resolve()


def coverage_summary_paths(runs_dir: Path) -> list[Path]:
    if not runs_dir.exists() or not runs_dir.is_dir():
        return []

    summaries = [
        summary
        for summary in runs_dir.glob("coverage_run_*/summary.json")
        if summary.is_file()
    ]
    return sorted(summaries, key=lambda path: path.stat().st_mtime, reverse=True)


def coverage_summary_error(summary_path: Path) -> str:
    try:
        with summary_path.open() as handle:
            summary = json.load(handle)
    except json.JSONDecodeError:
        return f"Coverage summary is not valid JSON: {summary_path}"

    if not isinstance(summary, dict):
        return f"Coverage summary has the wrong format: expected a JSON object in {summary_path}"

    if "images" not in summary or not isinstance(summary["images"], list):
        return f"Coverage summary is missing an images list: {summary_path}"
    for image in summary["images"]:
        if isinstance(image, str):
            continue
        if isinstance(image, dict) and isinstance(image.get("file"), str) and image["file"]:
            continue
        return f"Coverage summary has an invalid image entry: {summary_path}"

    return "Coverage summary could not be loaded."


def coverage_image_url(image: str, run_dir: Path, serving_root: Path) -> str:
    image_path = (run_dir / image).resolve()
    if is_relative_to(image_path, serving_root):
        return f"/coverage/{image_path.relative_to(serving_root).as_posix()}"
    return f"/coverage/{image}"


def coverage_image_payload(
    image: Any,
    run_dir: Path,
    serving_root: Path,
) -> dict[str, Any]:
    if isinstance(image, str):
        payload = {"file": image}
        image_file = image
    elif isinstance(image, dict) and isinstance(image.get("file"), str) and image["file"]:
        payload = dict(image)
        image_file = str(image["file"])
    else:
        raise HTTPException(status_code=422, detail="Coverage summary has an invalid image entry.")

    payload["url"] = coverage_image_url(image_file, run_dir, serving_root)
    return payload


def coverage_summary_payload(
    summary_path: Path,
    serving_root: Path,
) -> dict[str, Any]:
    try:
        with summary_path.open() as handle:
            summary = json.load(handle)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=422,
            detail=coverage_summary_error(summary_path),
        ) from None

    if (
        not isinstance(summary, dict)
        or "images" not in summary
        or not isinstance(summary["images"], list)
    ):
        raise HTTPException(status_code=422, detail=coverage_summary_error(summary_path))

    run_dir = summary_path.parent
    summary["images"] = [
        coverage_image_payload(image, run_dir, serving_root)
        for image in summary.get("images", [])
    ]
    summary["run_name"] = run_dir.name
    return summary


def coverage_status_payload(config: ServerConfig) -> dict[str, Any]:
    runs_dir = coverage_runs_dir(config)
    summaries = coverage_summary_paths(runs_dir)
    return {
        "enabled": True,
        "has_run": len(summaries) > 0,
        "path": str(runs_dir),
        "run_count": len(summaries),
    }


def latest_coverage_summary_payload(config: ServerConfig) -> dict[str, Any]:
    runs_dir = coverage_runs_dir(config)
    summaries = coverage_summary_paths(runs_dir)
    if not summaries:
        raise HTTPException(
            status_code=404,
            detail=f"No coverage runs found in: {runs_dir}",
        )

    return coverage_summary_payload(summaries[0], runs_dir)


def coverage_runs_payload(config: ServerConfig) -> dict[str, Any]:
    runs_dir = coverage_runs_dir(config)
    runs = [
        coverage_summary_payload(summary_path, runs_dir)
        for summary_path in coverage_summary_paths(runs_dir)
    ]
    return {
        "coverage_runs_dir": str(runs_dir),
        "runs": runs,
    }
