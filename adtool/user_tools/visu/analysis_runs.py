from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from .runtime import ServerConfig
from .server_support import is_relative_to


def analysis_runs_dir(config: ServerConfig) -> Path:
    return (config.discoveries.parent / "analysis_runs").resolve()


def analysis_summary_paths(runs_dir: Path) -> list[Path]:
    if not runs_dir.exists() or not runs_dir.is_dir():
        return []

    summaries = [
        summary
        for summary in runs_dir.glob("analysis_run_*/summary.json")
        if summary.is_file()
    ]
    return sorted(summaries, key=lambda path: path.stat().st_mtime, reverse=True)


def analysis_summary_error(summary_path: Path) -> str:
    try:
        with summary_path.open() as handle:
            summary = json.load(handle)
    except json.JSONDecodeError:
        return f"Analysis summary is not valid JSON: {summary_path}"

    if not isinstance(summary, dict):
        return f"Analysis summary has the wrong format: expected a JSON object in {summary_path}"

    if "modules" not in summary or "module_order" not in summary:
        return f"Analysis summary is missing modules: {summary_path}"
    for module_name in summary["module_order"]:
        module = summary["modules"].get(module_name, {})
        for image in module.get("images", []):
            if isinstance(image, dict) and isinstance(image.get("file"), str) and image["file"]:
                continue
            return f"Analysis summary has an invalid image entry: {summary_path}"

    return "Analysis summary could not be loaded."


def analysis_file_url(image: str, run_dir: Path, serving_root: Path) -> str:
    image_path = (run_dir / image).resolve()
    if is_relative_to(image_path, serving_root):
        return f"/analysis_files/{image_path.relative_to(serving_root).as_posix()}"
    return f"/analysis_files/{image}"


def analysis_image_payload(
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
        raise HTTPException(status_code=422, detail="Analysis summary has an invalid image entry.")

    payload["url"] = analysis_file_url(image_file, run_dir, serving_root)
    return payload


def analysis_summary_payload(
    summary_path: Path,
    serving_root: Path,
) -> dict[str, Any]:
    try:
        with summary_path.open() as handle:
            summary = json.load(handle)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=422,
            detail=analysis_summary_error(summary_path),
        ) from None

    if (
        not isinstance(summary, dict)
        or "modules" not in summary
        or "module_order" not in summary
    ):
        raise HTTPException(status_code=422, detail=analysis_summary_error(summary_path))

    run_dir = summary_path.parent
    for module_name in summary["module_order"]:
        module = summary["modules"][module_name]
        module["images"] = [
            analysis_image_payload(image, run_dir, serving_root)
            for image in module.get("images", [])
        ]
    summary["run_name"] = run_dir.name
    return summary


def analysis_status_payload(config: ServerConfig) -> dict[str, Any]:
    runs_dir = analysis_runs_dir(config)
    summaries = analysis_summary_paths(runs_dir)
    return {
        "enabled": True,
        "has_run": len(summaries) > 0,
        "path": str(runs_dir),
        "run_count": len(summaries),
    }


def latest_analysis_summary_payload(config: ServerConfig) -> dict[str, Any]:
    runs_dir = analysis_runs_dir(config)
    summaries = analysis_summary_paths(runs_dir)
    if not summaries:
        raise HTTPException(
            status_code=404,
            detail=f"No analysis runs found in: {runs_dir}",
        )

    return analysis_summary_payload(summaries[0], runs_dir)


def analysis_runs_payload(config: ServerConfig) -> dict[str, Any]:
    runs_dir = analysis_runs_dir(config)
    runs = [
        analysis_summary_payload(summary_path, runs_dir)
        for summary_path in analysis_summary_paths(runs_dir)
    ]
    return {
        "analysis_runs_dir": str(runs_dir),
        "runs": runs,
    }
