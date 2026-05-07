# embedded_systems: Add and Configure Modules

This example is fully path-configured: new components are loaded with `path` strings from JSON.
Core interference modules now live under `examples/embedded_systems/examples/core_interferences/`.

## 1) Add a new module

Place your class in the right folder (under
`examples/embedded_systems/examples/core_interferences/` for the core interference system):

- `generators/` for instruction generators
- `mutators/` for mutation strategies
- `mixers/` for parent-program mixing
- `goal_samplers/` for goal sampling
- `behavior_encoders/` for behavior embeddings
- `maps/` for parameter/behavior maps
- `simulator/` for simulator backends
- `simulator_runners/` for simulator loop runners

Then export it in that folder's `__init__.py`.

Example:

```python
from examples.embedded_systems.examples.core_interferences.mixers.my_new_mixer import MyNewMixer
```

## 2) Configure it in JSON

Use the class path in the core interference configs under
`examples/embedded_systems/examples/core_interferences/`.

Example snippets:

```json
{
  "system": {
    "config": {
      "simulator_config": {
        "path": "examples.embedded_systems.examples.core_interferences.simulator.MySimulatorBackend",
        "cycles": 512,
        "num_banks": 4,
        "num_addr": 96
      },
      "simulator_runner_config": {
        "path": "examples.embedded_systems.examples.core_interferences.simulator_runners.MyRunner"
      }
    }
  }
}
```

```json
{
  "explorer": {
    "config": {
      "mixer_config": {
        "path": "examples.embedded_systems.examples.core_interferences.mixers.MyNewMixer"
      },
      "behavior_map_config": {
        "goal_sampler_config": {
          "path": "examples.embedded_systems.goal_samplers.MyGoalSampler"
        },
        "behavior_encoder_config": {
          "path": "examples.embedded_systems.examples.core_interferences.behavior_encoders.MyEncoder"
        }
      },
      "parameter_map_config": {
        "generator_config": {
          "path": "examples.embedded_systems.examples.core_interferences.generators.MyGenerator"
        },
        "mutator_config": {
          "path": "examples.embedded_systems.examples.core_interferences.mutators.MyMutator"
        }
      }
    }
  }
}
```

## 3) Required interfaces (minimal)

- Generator: `generate(...) -> InstructionProgram`
- Mutator: `mutate(...) -> InstructionProgram`
- Mixer: `mix(sequences, max_cycle=...) -> InstructionProgram`
- Goal sampler: `sample(history, feature_size) -> np.ndarray`
- Behavior encoder: `encode(raw_output) -> np.ndarray`
- Simulator: `run(params) -> dict`
- Simulator runner: `run(params) -> dict`

See `types.py` for protocol contracts and
`examples/core_interferences/types.py` for the core interference payload types.
