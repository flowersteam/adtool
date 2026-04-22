# core_interference: Add and Configure Modules

This example is fully path-configured: new components are loaded with `path` strings from JSON.

## 1) Add a new module

Place your class in the right folder:

- `generators/` for instruction generators
- `mutators/` for mutation strategies
- `mixers/` for parent-program mixing
- `goal_samplers/` for goal sampling
- `behavior_encoders/` for behavior embeddings
- `maps/` for parameter/behavior maps
- `simulator_runners/` for simulator backend runners

Then export it in that folder's `__init__.py`.

Example:

```python
from examples.core_interference.mixers.my_new_mixer import MyNewMixer
```

## 2) Configure it in JSON

Use the class path in `core_interference_1.json`.

Example snippets:

```json
{
  "system": {
    "config": {
      "simulator_runner_config": {
        "path": "examples.core_interference.simulator_runners.MyRunner",
        "cycles": 512,
        "num_banks": 4,
        "num_addr": 96
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
        "path": "examples.core_interference.mixers.MyNewMixer"
      },
      "behavior_map_config": {
        "goal_sampler_config": {
          "path": "examples.core_interference.goal_samplers.MyGoalSampler"
        },
        "behavior_encoder_config": {
          "path": "examples.core_interference.behavior_encoders.MyEncoder"
        }
      },
      "parameter_map_config": {
        "generator_config": {
          "path": "examples.core_interference.generators.MyGenerator"
        },
        "mutator_config": {
          "path": "examples.core_interference.mutators.MyMutator"
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
- Simulator runner: `run(params) -> dict`

See `types.py` for protocol/type contracts.
