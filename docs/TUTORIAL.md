# Tutorial

This page is a very short guide to start a new project that uses `adtool` as an external library.
Run the commands from your own project folder.

## 1. Install and import

Use Python 3.9 to 3.12.

You can create a virtual environment in 3.12 with this command:
```bash
python3.12 -m venv .venv
```

```bash
pip install git+https://github.com/flowersteam/adtool
```

Optional, if you also want the visualization server:

```bash
pip install "adtool[visu] @ git+https://github.com/flowersteam/adtool"
```

Then you can import the main modules like so:

```python
from adtool.systems.System import System
from adtool.utils.expose_config.expose_config import expose
from adtool.examples.run import main
```

## 2. What you need to build

If a close example already exists in the library, start from it and only change the config or a few classes.

If your complex system does not already exist, you usually need to implement three pieces:

- `system`: runs your simulation or model.
  `map(data_dict)`: reads `data_dict["params"]`, runs one simulation, stores the result in `data_dict["output"]`, and returns the dict.
  `render(data_dict)`: makes an image or a video for visualization and returns it as bytes.
  Here's an example:
```python
from adtool.systems.System import System
class MySystem(System):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def map(self, data_dict):
        # Read the parameters chosen by the explorer
        # And run one simulation
        data_dict["output"] = run_system(data_dict["params"])
        return data_dict

    def render(self, data_dict):
        # Build one image or one video for the viewer
        image_bytes = render_output(data_dict["output"])
        return [(image_bytes, "png")]
```

- `parameter_map`: defines what can be explored.
  `sample()`: creates a random parameter (used for the initialisation).
  `mutate(parameter_dict)`: slightly changes an existing parameter set.
  `map(input, override_existing=True)`: puts the parameters in the shared dict under `"params"`.
  Here's an example:
```python
from adtool.maps.Map import Map
class MyParameterMap(Map):
    def __init__(self, system, premap_key="params"):
        super().__init__()
        self.premap_key = premap_key
        
    def sample(self):
        # Create one random parameter
        return {
            "dynamic_params": {
                "a": random_a(),
                "b": random_b(),
            }
        }

    def mutate(self, parameter_dict):
        # Slightly change an existing parameter set
        new_params = deepcopy(parameter_dict)
        new_params["dynamic_params"]["a"] = mutate_a(new_params["dynamic_params"]["a"])
        new_params["dynamic_params"]["b"] = mutate_b(new_params["dynamic_params"]["b"])
        return new_params

    def map(self, input, override_existing=True):
        # Put parameters in data_dict["params"]
        data_dict = deepcopy(input)
        # The following checks if we want to initialize 
        # or if there's an issue with the input, and thus sample randomly
        if self.premap_key not in data_dict or override_existing:
            data_dict[self.premap_key] = self.sample()
        return data_dict
```

- `behavior_map`: turns the raw system result into a small behavior vector.
  `map(input)`: reads `input["output"]`, computes a compact description, and writes it back in the dict.
  `sample()`: returns one target point in behavior space.
  Here's an example:
```python
from adtool.maps.Map import Map
class MyBehaviorMap(Map):
    def __init__(self, system, premap_key="output", postmap_key="output"):
        super().__init__()
        self.premap_key = premap_key
        self.postmap_key = postmap_key

    def map(self, input):
        # Read the raw system result
        data_dict = deepcopy(input)
        raw_output = data_dict[self.premap_key]

        # Keep the raw result and replace it with a smaller behavior vector
        data_dict["raw_" + self.premap_key] = raw_output
        data_dict[self.postmap_key] = compute_behavior_vector(raw_output)
        return data_dict

    def sample(self):
        # Return one target goal in behavior space
        # The target will be a vector of the behavior map
        new_target = generate_new_goal()
        return new_target
```

Keep the behavior vector simple at the start: a small `numpy` array or a short list of normalized numbers is usually enough.

All classes are loaded from dotted `path` strings in the JSON config, so keep file names, class names, and config paths aligned.

You can also modify the explorer if needed.
The default one is `adtool.explorers.IMGEPExplorer.IMGEPExplorer`, and it is enough for many projects.
If you want a different search strategy, set `"explorer.path"` to your own class.
In practice, the explorer must:

- create the first parameters with `bootstrap()`
- read one finished simulation with `map(system_output)`
- propose the next parameters to test

## 3. Simple folder structure

```text
my_adtool_project/
  config.json
  run.py
  my_project/
    __init__.py
    systems/
      __init__.py
      MySystem.py
      simulator/
        __init__.py
    behavior_map/
      __init__.py
      MyBehaviorMap.py
      encoder/
        __init__.py
      goal_sampler/
        __init__.py
    parameter_map/
      __init__.py
      MyParameterMap.py
      mutator/
        __init__.py
```

The `__init__.py` files make your folders importable.

Use your own package name in config paths. Example:
`my_project.systems.MySystem.MySystem`


## 4. Minimal config file

```json
{
  "experiment": {
    "config": {
      "save_location": "./runs/",
      "save_frequency": 1,
      "bootstrap_size": 1
    }
  },
  "system": {
    "path": "my_project.systems.MySystem.MySystem",
    "config": {}
  },
  "explorer": {
    "path": "adtool.explorers.IMGEPExplorer.IMGEPExplorer",
    "config": {
      "mutator": "specific",
      "equil_time": 1,
      "behavior_map": "my_project.behavior_map.MyBehaviorMap.MyBehaviorMap",
      "parameter_map": "my_project.parameter_map.MyParameterMap.MyParameterMap",
      "mutator_config": {},
      "behavior_map_config": {},
      "parameter_map_config": {}
    }
  },
  "logger_handlers": [],
  "callbacks": {
    "on_discovery": [
      {
        "path": "adtool.callbacks.on_discovery_callbacks.save_discovery_on_disk.SaveDiscoveryOnDisk",
        "config": {}
      }
    ]
  }
}
```

In the JSON file, most modules have two parts:

- `"path"`: which Python class to load
- `"config"`: the parameters given to this class when it is created

So, for example:

- `system.config` contains the settings of your system
- `explorer.config` contains the settings of the explorer
- `callbacks.on_discovery[i].config` contains the settings of one callback

Some modules also create sub-modules internally.
This is why you may see fields such as `parameter_map_config`,
`behavior_map_config`, or `mutator_config` inside `explorer.config`.
These are simply the configuration dictionaries passed to these inner modules.

An empty `"config": {}` just means: "create this module with its default settings".

`save_location` is where `adtool` will create the `discoveries/` folder.
If the `save_location` already has experiments inside, the algorithm will try to reuse them, so keep the same configuration.
Keep the save callback at the start: it writes the files used by the viewer.

## 5. New configuration parameters

To make a module accept new configuration parameters, you must change the Python code of this module.
In practice, the values written inside `"config": {...}` are passed when the class is created.
So if you want a new key in JSON, your class must be able to receive and use it.

The cleanest and recommended way is to define a small Pydantic config class and use `@expose`:

```python
from pydantic import BaseModel, Field
from adtool.systems.System import System
from adtool.utils.expose_config.expose_config import expose

class MySystemConfig(BaseModel):
    size: int = Field(64, ge=8)
    num_steps_simu: int = Field(100, ge=1)
    noise: float = Field(0.1, ge=0.0, le=1.0)

@expose
class MySystem(System):
    config = MySystemConfig

    def __init__(self, size=64, num_steps_simu=100, noise=0.1, *args, **kwargs):
        super().__init__()
        self.size = size
        self.num_steps_simu = num_steps_simu
        self.noise = noise

    def map(self, data_dict):
        data_dict["output"] = run_system(
            params=data_dict["params"],
            size=self.size,
            num_steps_simu=self.num_steps_simu,
            noise=self.noise,
        )
        return data_dict
```

Then you can write:

```json
"system": {
  "path": "my_project.systems.MySystem.MySystem",
  "config": {
    "size": 128,
    "num_steps_simu": 300,
    "noise": 0.05
  }
}
```

The same idea applies to `parameter_map`, `behavior_map`, callbacks, and any custom explorer.

For more detail, the best next step is to read a few real configs in [`examples/`](/home/arthur/Documents/INRIA/codes/adtool/examples) and the short architecture note in [docs/ARCHITECTURE.md](/home/arthur/Documents/INRIA/codes/adtool/docs/ARCHITECTURE.md).


## 6. Run the experiment

The easiest way is to use the runner already provided by the library:

```bash
python -m adtool.examples.run --config_file config.json --nb_iterations 40
```
or helper
```bash
python -m adtool.examples.run -h
```

If you want your own small entrypoint, keep it very thin and reuse the same runner:

```python
from adtool.examples.run import main

if __name__ == "__main__":
    main()
```

Then run:

```bash
python run.py --config_file config.json --nb_iterations 40
```
or helper
```bash
python run.py -h
```

This is a good starting point because later you can tweak this runner to launch several experiments, several seeds, or extra logic around the same base command.

To open the visualization after a run:

```bash
python -m adtool.examples.visu.server --discoveries ./runs/discoveries
```
or helper
```bash
python -m adtool.examples.visu.server -h
```

## 7. Good first rule

Before writing code, identify which sub-operators you will need.
Then create the nested folders for them directly, for example:

- `systems/`
- `systems/simulator/`
- `systems/runner/`
- `behavior_map/encoder/`
- `behavior_map/goal_sampler/`
- `parameter_map/mutator/`

This usually keeps the project easier to read and easier to edit from the start.
