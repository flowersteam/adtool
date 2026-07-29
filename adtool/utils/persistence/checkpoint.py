"""Filesystem checkpoints for experiment Leaves.

The storage interface is intentionally independent from the old Locator API.
Future stores can implement :class:`CheckpointStore` without changing explorer
or pipeline history code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import tempfile
from typing import Any
from uuid import uuid4

from adtool.utils.persistence.history import HistoryStore
from adtool.utils.factory import class_path_of, resolve_dotted_object
from adtool.utils.leaf.Leaf import Leaf


@dataclass(frozen=True)
class CheckpointRef:
    """A completed filesystem checkpoint."""

    path: Path
    manifest: dict[str, Any]

    def __str__(self) -> str:
        return str(self.path)


class CheckpointStore(ABC):
    """Persistence boundary for experiment checkpoints."""

    @abstractmethod
    def save(self, pipeline: Leaf) -> CheckpointRef:
        raise NotImplementedError

    @abstractmethod
    def load(self, checkpoint: str | Path) -> Leaf:
        raise NotImplementedError


class FileCheckpointStore(CheckpointStore):
    """Write one pickle state file per Leaf component."""

    FORMAT_VERSION = "v0"

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    def save(self, pipeline: Leaf) -> CheckpointRef:
        if not hasattr(pipeline, "_explorer"):
            raise TypeError("FileCheckpointStore expects an ExperimentPipeline-like Leaf")

        history = getattr(pipeline._explorer, "history", None)
        if not isinstance(history, HistoryStore):
            raise TypeError("Explorer does not expose a HistoryStore")

        system_name = self._system_name(getattr(pipeline, "config", {}))
        config_hash = self._config_hash(getattr(pipeline, "config", {}))
        branch_id = getattr(pipeline, "branch_id", None) or uuid4().hex
        step = int(getattr(pipeline, "run_idx", 0))
        container = self.root / "checkpoints"
        container.mkdir(parents=True, exist_ok=True)
        checkpoint_name = f"step-{step:08d}-cfg-{config_hash[:12]}-branch-{branch_id[:8]}"
        destination = container / checkpoint_name
        if destination.exists():
            raise FileExistsError(f"Checkpoint already exists: {destination}")

        temporary = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=container))
        try:
            components = self._write_components(temporary, pipeline)
            history_files, history_file_counts = self._write_history(
                temporary, history, checkpoint_name=checkpoint_name
            )
            manifest = {
                "format_version": self.FORMAT_VERSION,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "system_name": system_name,
                "config_hash": config_hash,
                "step": step,
                "branch_id": branch_id,
                "parent_checkpoint": str(history.head) if history.head else None,
                "components": components,
                "history_files": history_files,
                "history_file_counts": history_file_counts,
            }
            with (temporary / "manifest.json").open("w") as file:
                json.dump(manifest, file, indent=2, sort_keys=True)
            os.replace(temporary, destination)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

        history.commit_checkpoint(
            destination,
            history_files=history_files,
            history_file_counts=history_file_counts,
        )
        return CheckpointRef(path=destination, manifest=manifest)

    def load(self, checkpoint: str | Path) -> Leaf:
        checkpoint_dir = self._resolve_checkpoint_dir(checkpoint)
        manifest_path = checkpoint_dir / "manifest.json"
        with manifest_path.open() as file:
            manifest = json.load(file)
        if manifest.get("format_version") != self.FORMAT_VERSION:
            raise ValueError(f"Unsupported checkpoint format: {manifest.get('format_version')}")

        components: dict[str, Leaf] = {}
        metadata = manifest["components"]
        for item in metadata:
            component_path = item["path"]
            cls = resolve_dotted_object(item["class_path"], object_name="checkpoint component")
            with (checkpoint_dir / item["file"]).open("rb") as file:
                state = pickle.load(file)
            obj = cls.__new__(cls)
            Leaf._default_leaf_init(obj)
            obj.__dict__.update(state)
            obj._set_attr_override("_modules", {})
            obj._set_attr_override("_container_ptr", None)
            components[component_path] = obj

        for item in sorted(metadata, key=lambda entry: entry["path"].count(".")):
            path = item["path"]
            parent_path = item.get("parent")
            if parent_path is None:
                continue
            parent = components[parent_path]
            child = components[path]
            parent._bind_submodule_to_self(item["attribute"], child)

        pipeline = components["pipeline"]
        explorer = getattr(pipeline, "_explorer")
        feature_key = getattr(explorer, "premap_key", "output")
        payload_key = getattr(explorer, "postmap_key", "params")
        experiment_config = getattr(pipeline, "config", {}).get("experiment", {}).get(
            "config", {}
        )
        cache_size = int(
            experiment_config.get(
                "discoveries_cache_size", getattr(pipeline, "save_frequency", 1)
            )
        )
        history = HistoryStore.from_checkpoint(
            checkpoint_dir,
            feature_key=feature_key,
            payload_key=payload_key,
            cache_size=cache_size,
        )
        explorer.history = history
        if hasattr(explorer, "restore_checkpoint_runtime"):
            explorer.restore_checkpoint_runtime()
        if hasattr(pipeline, "restore_checkpoint_runtime"):
            pipeline.restore_checkpoint_runtime(CheckpointRef(checkpoint_dir, manifest))
        return pipeline

    def _resolve_checkpoint_dir(self, checkpoint: str | Path) -> Path:
        """Resolve an explicit path or a checkpoint folder name.

        A bare folder name is looked up directly below
        ``<save_location>/checkpoints``.  For compatibility, the resolver also
        recognizes checkpoints written by the former ``<system>/`` layout.
        """
        requested = Path(checkpoint)
        if requested.is_absolute():
            candidates = [requested]
        else:
            candidates = [
                self.root / requested,
                self.root / "checkpoints" / requested,
            ]
            if requested.parent == Path("."):
                candidates.extend(
                    (self.root / "checkpoints").glob(f"*/{requested.name}")
                )

        matches = sorted(
            {
                candidate.resolve()
                for candidate in candidates
                if (candidate / "manifest.json").is_file()
            },
            key=str,
        )
        if not matches:
            raise FileNotFoundError(
                "Could not find checkpoint {!r}. Use its folder name from "
                "<save_location>/checkpoints/, or pass an explicit path."
                .format(str(checkpoint))
            )
        if len(matches) > 1:
            locations = ", ".join(str(match) for match in matches)
            raise ValueError(
                f"Checkpoint folder name {checkpoint!r} is ambiguous: {locations}. "
                "Use a path relative to save_location instead."
            )
        return matches[0]

    def _write_components(self, directory: Path, root: Leaf) -> list[dict[str, str | None]]:
        components: list[dict[str, str | None]] = []
        for path, leaf, parent_path, attribute in self._walk_leaves(root):
            filename = path.replace(".", "__") + ".pickle"
            state = self._checkpoint_state(leaf)
            with (directory / filename).open("wb") as file:
                pickle.dump(state, file, protocol=pickle.HIGHEST_PROTOCOL)
            components.append(
                {
                    "path": path,
                    "class_path": class_path_of(leaf.__class__),
                    "file": filename,
                    "parent": parent_path,
                    "attribute": attribute,
                }
            )
        return components

    @staticmethod
    def _write_history(
        directory: Path, history: HistoryStore, *, checkpoint_name: str
    ) -> tuple[list[str], dict[str, int]]:
        return history.write_checkpoint_history(
            directory, checkpoint_name=checkpoint_name
        )

    @staticmethod
    def _walk_leaves(root: Leaf):
        stack: list[tuple[str, Leaf, str | None, str | None]] = [("pipeline", root, None, None)]
        while stack:
            path, leaf, parent, attribute = stack.pop()
            yield path, leaf, parent, attribute
            for name, child in reversed(list(leaf._modules.items())):
                if not isinstance(child, Leaf):
                    raise TypeError(f"Leaf module {path}.{name} is not a Leaf")
                display_name = name.lstrip("_") or name
                stack.append((f"{path}.{display_name}", child, path, name))

    @staticmethod
    def _checkpoint_state(leaf: Leaf) -> dict[str, Any]:
        if hasattr(leaf, "checkpoint_state"):
            return leaf.checkpoint_state()
        state = dict(leaf.__dict__)
        for key in ("_modules", "_container_ptr", "logger"):
            state.pop(key, None)
        for key, value in list(state.items()):
            if isinstance(value, HistoryStore):
                state.pop(key)
        return state

    @staticmethod
    def _system_name(config: dict[str, Any]) -> str:
        path = config.get("system", {}).get("path", "system")
        name = str(path).split(".")[-1]
        return "".join(char if char.isalnum() or char in "-_" else "_" for char in name)

    @staticmethod
    def _config_hash(config: dict[str, Any]) -> str:
        normalized = deepcopy(config)
        experiment = normalized.get("experiment", {}).get("config", {})
        experiment.pop("save_location", None)
        experiment.pop("resume_checkpoint", None)
        encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(encoded.encode()).hexdigest()
