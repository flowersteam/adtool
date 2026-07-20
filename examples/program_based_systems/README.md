# Program-Based Systems: Add and Configure Modules

This example is fully path-configured: new components are loaded with `path` strings from JSON.
Core interference modules now live under `examples/program_based_systems/examples/core_interferences/`.

## 1) Add a new module

Place your class in the right folder (under
`examples/program_based_systems/examples/core_interferences/` for the core interference system):

- `parameter_map/` for parameter maps
- `parameter_map/mutator/` for generators, mutation strategies, and parent-program mixing
- `behavior_map/` for behavior maps
- `behavior_map/encoder/` for behavior embeddings
- `behavior_map/goal_sampler/` for goal sampling
- `systems/simulator/` for simulator backends
- `systems/runner/` for simulator loop runners

Then export it in that folder's `__init__.py`.

Example:

```python
from examples.program_based_systems.examples.core_interferences.parameter_map.mutator.my_new_mixer import MyNewMixer
```

## 2) Configure it in JSON

Use the class path in the core interference configs under
`examples/program_based_systems/examples/core_interferences/`.

Example snippets:

```json
{
  "system": {
    "config": {
      "simulator": {
        "path": "examples.program_based_systems.examples.core_interferences.systems.simulator.MySimulatorBackend",
        "config": {
          "cycles": 512,
          "num_banks": 4,
          "num_addr": 96
        }
      },
      "simulator_runner": {
        "path": "examples.program_based_systems.examples.core_interferences.systems.runner.MyRunner",
        "config": {}
      }
    }
  }
}
```

```json
{
  "explorer": {
    "config": {
      "behavior_map": {
        "path": "examples.program_based_systems.examples.core_interferences.behavior_map.InterferenceBehaviorMap",
        "config": {
          "goal_sampler": {
            "path": "examples.program_based_systems.behavior_map.goal_sampler.MyGoalSampler",
            "config": {}
          },
          "behavior_encoder": {
            "path": "examples.program_based_systems.examples.core_interferences.behavior_map.encoder.MyEncoder",
            "config": {}
          }
        }
      },
      "parameter_map": {
        "path": "examples.program_based_systems.examples.core_interferences.parameter_map.InterferenceParameterMap",
        "config": {
          "mixer": {
            "path": "examples.program_based_systems.examples.core_interferences.parameter_map.mutator.MyNewMixer",
            "config": {}
          },
          "generator": {
            "path": "examples.program_based_systems.examples.core_interferences.parameter_map.mutator.MyGenerator",
            "config": {}
          },
          "mutator": {
            "path": "examples.program_based_systems.examples.core_interferences.parameter_map.mutator.MyMutator",
            "config": {}
          }
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
