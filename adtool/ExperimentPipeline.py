"""Experiment execution with explicit file checkpoints and chunked history."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import traceback
from typing import Callable, Iterable, List
from uuid import uuid4

from adtool.utils.persistence.checkpoint import CheckpointRef, FileCheckpointStore
from adtool.utils.persistence.history import HistoryStore
from adtool.utils.interaction.experiment_control import (
    read_experiment_control,
    wait_if_experiment_paused,
)
from adtool.utils.leaf.Leaf import Leaf
from adtool.utils.leaf.locators.locators import BlobLocator

def get_render_every(config: dict) -> int:
    render_every = int(config["experiment"]["config"].get("render_every", 1))
    if render_every < 0:
        raise ValueError("experiment.config.render_every must be >= 0")
    return render_every


class ExperimentPipeline(Leaf):
    """
    Pipeline of an automated discovery experiment.

    An experiment is at least constituted of a system and an explorer.

    Callbacks are dispatched for discovery, checkpoint save, save completion,
    normal completion, and errors.

    In order to monitor the experiment, you must provide **callbacks**.
    Please see: `callbacks.base_callback.BaseCallback`.
    """

    def __init__(
        self,
        config: dict | None = None,
        experiment_id: int = 0,
        seed: int = 0,
        system=None,
        explorer=None,
        save_frequency: int = 100,
        on_discovery_callbacks: List[Callable] | None = None,
        on_save_finished_callbacks: List[Callable] | None = None,
        on_finished_callbacks: List[Callable] | None = None,
        on_save_callbacks: List[Callable] | None = None,
        on_error_callbacks: List[Callable] | None = None,
        logger=None,
        resource_uri: str = "",
        discovery_saving_keys: List[str] | None = None,
    ) -> None:
        super().__init__()
        self.locator = BlobLocator(resource_uri)
        self.run_idx = 0
        self.experiment_id = experiment_id
        self.seed = seed
        self.discovery_saving_keys = list(discovery_saving_keys or [])
        self._pending_trial: dict | None = None
        self._checkpoint_ref: CheckpointRef | None = None

        self._system = system
        self._explorer = explorer

        self._bind_runtime(
            config=config or {},
            resource_uri=resource_uri,
            logger=logger,
            callbacks={
                "on_discovery": on_discovery_callbacks or [],
                "on_save_finished": on_save_finished_callbacks or [],
                "on_finished": on_finished_callbacks or [],
                "on_save": on_save_callbacks or [],
                "on_error": on_error_callbacks or [],
            },
            save_frequency=save_frequency,
        )

    def _raise_callbacks(self, callbacks: Iterable[Callable], **kwargs) -> None:
        for callback in callbacks:
            callback(**kwargs)

    def _log(self, level: str, message: str) -> None:
        if self.logger is not None:
            getattr(self.logger, level)(message)

    def _should_render(self, render_every: int) -> bool:
        return render_every > 0 and self.run_idx % render_every == 0

    def _history(self) -> HistoryStore:
        history = getattr(self._explorer, "history", None)
        if not isinstance(history, HistoryStore):
            raise TypeError("Explorer must expose a HistoryStore as 'history'")
        return history

    def _configure_history_store(self) -> None:
        if self._explorer is None or self._checkpoint_store is None:
            return
        history = self._history()
        history.set_logger(self.logger)
        experiment_config = self.config.get("experiment", {}).get("config", {})
        discoveries_cache_size = int(
            experiment_config.get("discoveries_cache_size", self.save_frequency)
        )
        history_lookback_length = int(
            experiment_config.get("history_lookback_length", -1)
        )
        if discoveries_cache_size < -1:
            raise ValueError("experiment.config.discoveries_cache_size must be >= -1")
        if history_lookback_length < -1:
            raise ValueError(
                "experiment.config.history_lookback_length must be >= -1"
            )
        history.cache_size = discoveries_cache_size
        if discoveries_cache_size == 0:
            self._log(
                "warning",
                "[HISTORY] - discoveries_cache_size=0 disables the RAM history cache; "
                "history retrieval may read checkpoint chunks from disk.",
            )
        elif discoveries_cache_size == -1:
            self._log(
                "warning",
                "[HISTORY] - discoveries_cache_size=-1 keeps complete history in RAM; "
                "memory usage grows with every discovery.",
            )
        self._explorer.history_lookback_length = history_lookback_length

    def _bind_runtime(
        self,
        *,
        config: dict,
        resource_uri: str,
        logger,
        callbacks: dict[str, list[Callable]],
        save_frequency: int,
    ) -> None:
        """Bind runtime-only dependencies for a new or restored pipeline."""
        save_frequency = int(save_frequency)
        if save_frequency < 1:
            raise ValueError("save_frequency must be >= 1")
        self.config = config
        self.resource_uri = resource_uri
        self.locator.resource_uri = resource_uri
        self.save_frequency = save_frequency
        self.logger = logger
        self._on_discovery_callbacks = list(callbacks.get("on_discovery", []))
        self._on_save_finished_callbacks = list(callbacks.get("on_save_finished", []))
        self._on_finished_callbacks = list(callbacks.get("on_finished", []))
        self._on_save_callbacks = list(callbacks.get("on_save", []))
        self._on_error_callbacks = list(callbacks.get("on_error", []))
        self._checkpoint_store = FileCheckpointStore(resource_uri) if resource_uri else None
        self.branch_id = uuid4().hex
        self._configure_history_store()

    def configure_runtime(
        self,
        *,
        config: dict,
        resource_uri: str,
        logger,
        callbacks: dict[str, list[Callable]],
    ) -> None:
        """Reattach process-local state after loading a checkpoint."""
        self._bind_runtime(
            config=config,
            resource_uri=resource_uri,
            logger=logger,
            callbacks=callbacks,
            save_frequency=config["experiment"]["config"]["save_frequency"],
        )

    def run(self, n_exploration_runs: int):
        if n_exploration_runs < 0:
            raise ValueError("n_exploration_runs must be >= 0")
        if self._checkpoint_store is None:
            self._checkpoint_store = FileCheckpointStore(self.resource_uri)
            self._configure_history_store()

        control_dir = Path(self.resource_uri)
        discoveries_dir = control_dir / "discoveries"
        discoveries_dir.mkdir(parents=True, exist_ok=True)
        try:
            if self._pending_trial is not None:
                data_dict = deepcopy(self._pending_trial)
            else:
                data_dict = self._explorer.bootstrap()

            bootstrap_size = int(self.config["experiment"]["config"]["bootstrap_size"])
            render_every = get_render_every(self.config)
            final_run_idx = self.run_idx + n_exploration_runs

            while self.run_idx < final_run_idx:
                wait_if_experiment_paused(str(control_dir))
                goal_targeting = read_experiment_control(str(control_dir)).get(
                    "goal_targeting", {}
                ).get("resolved")
                target_path = discoveries_dir / "target.json"
                if target_path.is_file():
                    with target_path.open() as file:
                        data_dict["target"] = json.load(file)["target"]

                if self.run_idx < bootstrap_size:
                    data_dict = self._explorer.bootstrap()

                data_dict = self._system.map(data_dict)
                if goal_targeting is None:
                    data_dict.pop("goal_targeting", None)
                else:
                    data_dict["goal_targeting"] = goal_targeting

                rendered_outputs = (
                    self._system.render(data_dict) if self._should_render(render_every) else None
                )
                data_dict = self._explorer.map(data_dict)
                discovery = self._explorer.read_last_discovery()
                discovery_to_save = deepcopy(discovery)
                if self.discovery_saving_keys:
                    discovery_to_save = {
                        key: value
                        for key, value in discovery_to_save.items()
                        if key in self.discovery_saving_keys
                    }
                metadata = dict(discovery_to_save.get("metadata", {}))
                metadata.update(
                    {
                        "branch_id": self.branch_id,
                        "parent_checkpoint": str(self._checkpoint_ref.path)
                        if self._checkpoint_ref
                        else None,
                    }
                )
                discovery_to_save["metadata"] = metadata

                self._raise_callbacks(
                    self._on_discovery_callbacks,
                    config=self.config,
                    resource_uri=self.resource_uri,
                    run_idx=self.run_idx,
                    experiment_id=self.experiment_id,
                    seed=self.seed,
                    discovery=discovery_to_save,
                    rendered_outputs=rendered_outputs,
                )
                phase = "bootstrap" if self.run_idx < bootstrap_size else "exploration"
                self._log(
                    "info",
                    "[DISCOVERY] "
                    f"experiment={self.experiment_id} | "
                    f"step={self.run_idx + 1} | phase={phase} | "
                    f"branch={self.branch_id[:8]}",
                )

                self.run_idx += 1
                self._pending_trial = deepcopy(data_dict)
                if self.run_idx % self.save_frequency == 0 or self.run_idx == final_run_idx:
                    self.save(resource_uri=self.resource_uri)

        except Exception:
            message = "error in experiment {} self.run_idx {} seed {} = {}".format(
                self.experiment_id, self.run_idx, self.seed, traceback.format_exc()
            )
            self._log("error", "[ERROR] - " + message[:8000])
            self._raise_callbacks(
                self._on_error_callbacks,
                run_idx=self.run_idx,
                seed=self.seed,
                experiment_id=self.experiment_id,
            )
            raise

        self._log(
            "info",
            "[FINISHED] "
            f"experiment={self.experiment_id} | seed={self.seed} | "
            f"steps={self.run_idx} | branch={self.branch_id[:8]}",
        )
        self._raise_callbacks(
            self._on_finished_callbacks,
            run_idx=self.run_idx,
            seed=self.seed,
            experiment_id=self.experiment_id,
        )

    def save(self, resource_uri: str):
        if self._checkpoint_store is None:
            raise RuntimeError("Checkpoint store is not configured")
        self._checkpoint_ref = self._checkpoint_store.save(self)
        self._raise_callbacks(
            self._on_save_callbacks,
            experiment_id=self.experiment_id,
            seed=self.seed,
            module_to_save=self,
            resource_uri=resource_uri,
        )
        self._raise_callbacks(
            self._on_save_finished_callbacks,
            uid=str(self._checkpoint_ref.path),
            report_dir=resource_uri,
            experiment_id=self.experiment_id,
            seed=self.seed,
            run_idx=self.run_idx,
        )
        self._log(
            "info",
            "[CHECKPOINT] "
            f"experiment={self.experiment_id} | seed={self.seed} | "
            f"step={self.run_idx} | branch={self.branch_id[:8]} | "
            f"folder={self._checkpoint_ref.path.name}",
        )
        return self._checkpoint_ref

    def checkpoint_state(self) -> dict:
        state = super().checkpoint_state()
        for name in (
            "_on_discovery_callbacks",
            "_on_save_finished_callbacks",
            "_on_finished_callbacks",
            "_on_error_callbacks",
            "_on_save_callbacks",
            "_checkpoint_store",
            "_checkpoint_ref",
        ):
            state.pop(name, None)
        return state

    def restore_checkpoint_runtime(self, checkpoint: CheckpointRef) -> None:
        self._checkpoint_ref = checkpoint
