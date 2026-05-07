from typing_extensions import Any, Dict, List, Protocol

import numpy as np


class GoalSampler(Protocol):
    def sample(
        self,
        history: List[np.ndarray],
        feature_size: int | None,
        **kwargs: Any,
    ) -> np.ndarray:
        ...


class ProgramMixer(Protocol):
    def mix(
        self,
        sequences: List[Any],
        *,
        max_cycle: int,
    ) -> Any:
        ...


class ProgramMutator(Protocol):
    def mutate(
        self,
        instructions: Any,
        *,
        max_cycle: int,
        min_address: int,
        max_address: int,
        num_instructions: int,
    ) -> Any:
        ...


class ProgramGenerator(Protocol):
    def generate(
        self,
        *,
        num_instructions: int,
        max_cycle: int,
        min_address: int,
        max_address: int,
    ) -> Any:
        ...


class BehaviorEncoder(Protocol):
    def encode(self, raw_output: Dict[str, Any]) -> np.ndarray:
        ...


class Simulator(Protocol):
    ...


class SimulatorRunner(Protocol):
    def run(self, params: Any) -> Any:
        ...