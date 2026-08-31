# Analysis Modules

## What This Layer Is

The analysis module system is the offline post-processing layer of the library.

It runs on saved `discoveries/*/discovery.json` files after or alongside an exploration run. Unlike the explorer itself, an analysis module is allowed to use the full saved dataset at once. This is why analysis is meant for offline metrics, comparisons, summaries, plots, and batch post-processing.

Analysis modules operate on the saved numeric discovery payloads, not on rendered preview media. This matters when an experiment uses `experiment.config.render_every` with `0` or a sparse cadence: analysis still works because `discovery.json` is always saved even when `.png` or `.mp4` outputs are skipped.

Typical uses are:

- compare one discovery folder against one or more other folders,
- compute progression curves over discovery order,
- generate plots from all discoveries at once,
- add experiment-specific metrics without touching the central analysis runner or analysis UI.

Related saved files:

- discoveries live under `<save_location>/discoveries/`
- the shared run config is saved at `<save_location>/discoveries/config.json`
- rendered media inside each discovery folder is optional and may be absent when rendering is disabled or throttled

Analysis can be launched in two ways:

- from the command line with `python -m adtool.runners.run_analysis ...`,
- from the `Analysis` page of the visualization UI.

The UI does not change the execution model: it still launches offline analysis jobs on saved discoveries.

## How It Works

The entrypoint is `run_analysis`, exposed through:

- [adtool/runners/run_analysis.py](../adtool/runners/run_analysis.py)
- `adtool.user_tools.analysis_metrics.analysis_run.run_analysis`

Analysis is driven by a config file with a top-level `checkpoint_name` and `analysis_modules` list:

```json
{
  "checkpoint_name": null,
  "analysis_modules": [
    {
      "path": "adtool.user_tools.analysis_metrics.comparison_1d.Comparison1DModule",
      "config": {}
    },
    {
      "path": "examples.program_based_systems.examples.core_interferences.analysis_modules.mutual_miss_heatmap.MutualMissHeatmapModule",
      "config": {}
    }
  ]
}
```

Each entry is loaded independently, in order. The module order in this list is also the display order on the analysis page.

`checkpoint_name` selects the experiment branch to analyse. Leave it as `null`
(or omit it) to use the most recent checkpoint. To select a specific branch,
replace `null` with its checkpoint folder name, such as
`"step-00000300-cfg-dee0cab652c2-branch-0a76edac"`. Analysis includes that
checkpoint and its parents only; sibling and child branches are excluded.

Every discovery input accepts one of three directory forms:

- a directory containing saved `discovery.json` files,
- an experiment root containing `checkpoints/`, or
- a checkpoint directory containing `manifest.json`.

Checkpoint inputs are reconstructed cumulatively from the oldest ancestor to
the selected checkpoint. Checkpoint chunks with the same `branch_id` form one
plot series and use a stable color derived from that branch ID; saving another
checkpoint on the same branch does not change color. Ancestor branches use
`<label> ancestor N`, where ancestor 1 is closest to step 0, and shared
checkpoint chunks are included only once in 1D and 2D comparisons.

Coverage progression is computed over each complete selected path from step 0
to its selected checkpoint. Its line changes color only where `branch_id`
changes, and a colored point marks every checkpoint step along the path.

Analysis accepts one or more equal discovery inputs. If multiple inputs resolve
to the same selected checkpoint, that checkpoint is included only once, using
the first input's source and label.

The optional `checkpoint_name` setting applies to experiment-root inputs. An
explicit checkpoint path always selects itself.

Each module receives:

- `datasets`: loaded discovery sets,
- `labels`: dataset labels,
- `run_dir`: the output directory for generated images and summary data.

Each module returns a generic payload with at least:

- `title`
- `images`

The analysis UI renders modules generically from that payload. There is no module-specific frontend contract.

## Built-In Modules

The library currently ships with these built-in analysis modules:

- `Comparison1DModule`
  - applies a projection,
  - compares one-dimensional distributions across datasets,
  - generates density plots.
- `Comparison2DModule`
  - applies a projection,
  - compares selected dimension pairs across datasets,
  - generates 2D scatter plots.
- `SpaceCoverageModule`
  - applies a projection,
  - orders discoveries by `metadata.run_idx`,
  - computes progression of a coverage metric over discovery order.

There is also an experiment-local example custom module in core interferences:

- [mutual_miss_heatmap.py](../examples/program_based_systems/examples/core_interferences/analysis_modules/mutual_miss_heatmap.py)

That example shows the intended extension model:

- shared generic plumbing stays in `adtool.user_tools.analysis_metrics`,
- experiment-specific metrics stay next to the system that owns them.

### Using Built-In Modules

The built-in modules are used exactly like custom modules: add them to the `analysis_modules` list with their dotted path and module-specific config.

Minimal `Comparison1DModule` example:

```json
{
  "analysis_modules": [
    {
      "path": "adtool.user_tools.analysis_metrics.comparison_1d.Comparison1DModule",
      "config": {
        "projection": {
          "path": "examples.program_based_systems.examples.core_interferences.helpers.coverage_pretreatment.compact_interference_metrics",
          "config": {}
        },
        "dimensions": "all",
        "plot": {
          "points": 512,
          "format": "png",
          "color_a": "#4c78a8",
          "color_b": "#f58518",
          "alpha": 0.35,
          "line_width": 2.0,
          "figsize": [7.0, 4.0]
        }
      }
    }
  ]
}
```

Minimal `Comparison2DModule` example:

```json
{
  "analysis_modules": [
    {
      "path": "adtool.user_tools.analysis_metrics.comparison_2d.Comparison2DModule",
      "config": {
        "projection": {
          "path": "examples.program_based_systems.examples.core_interferences.helpers.coverage_pretreatment.compact_interference_metrics",
          "config": {}
        },
        "pairs": [[0, 12], [2, 12]],
        "plot": {
          "format": "png",
          "color_a": "#4c78a8",
          "color_b": "#f58518",
          "max_opacity": 0.8,
          "figsize": [7.0, 4.0]
        }
      }
    }
  ]
}
```

For a 2D comparison, repeated coordinates within one dataset are aggregated
into one marker. Its opacity is proportional to its occurrence count in that
dataset and never exceeds `max_opacity` (at most `0.8`), so overlapping
datasets remain visible.

Minimal `SpaceCoverageModule` example:

```json
{
  "analysis_modules": [
    {
      "path": "adtool.user_tools.analysis_metrics.space_coverage.SpaceCoverageModule",
      "config": {
        "projection": {
          "path": "examples.program_based_systems.examples.core_interferences.helpers.coverage_pretreatment.compact_interference_metrics",
          "config": {}
        },
        "metric": {
          "path": "examples.program_based_systems.examples.core_interferences.behavior_map.space_coverage.grid_space_coverage_metric.GridSpaceCoverageMetric",
          "config": {
            "dimensions": [0, 1, 2],
            "boundaries": [[-25, 25], [-25, 25], [-25, 25]],
            "bins_per_dimension": [8, 8, 8],
            "title": "Coverage progression"
          }
        },
        "plot": {
          "format": "png",
          "color_a": "#4c78a8",
          "color_b": "#f58518",
          "line_width": 2.0,
          "figsize": [7.0, 4.0]
        }
      }
    }
  ]
}
```

You can combine several built-in modules and custom modules in the same file. A full real example is available in:

- [core_interference_analysis.json](../examples/program_based_systems/examples/core_interferences/core_interference_analysis.json)

## Add a New Analysis Module

Adding a new metric is intentionally small:

1. Create a Python class inheriting `AnalysisModule`.
2. Implement `module_id`.
3. Implement `run(datasets, labels, run_dir)`.
4. Write your images into `run_dir`.
5. Return a generic module payload.
6. Reference the class from `analysis_modules` in the config.

Minimal template:

```python
from pathlib import Path

from adtool.user_tools.analysis_metrics.shared import AnalysisImage, AnalysisModule


class MyAnalysisModule(AnalysisModule):
    module_id = "my_metric"

    def run(self, datasets, labels, run_dir: Path) -> dict:
        image_name = "my_metric.png"

        # Compute your metric from all saved discoveries here.
        # datasets[i].payloads contains the loaded discovery.json payloads.

        output_path = run_dir / image_name
        # Write the figure to output_path.

        return {
            "title": "My metric",
            "images": [
                AnalysisImage(
                    file=image_name,
                    title="My metric",
                    plot_type="custom",
                    dimensions=[],
                    bounds=[],
                ).to_payload()
            ],
            "summary": ["1 graph"],
        }
```

Minimal config entry:

```json
{
  "analysis_modules": [
    {
      "path": "adtool.examples.my_system.analysis_modules.MyAnalysisModule",
      "config": {}
    }
  ]
}
```

## Run Analysis From The CLI

Example:

```bash
python -m adtool.runners.run_analysis \
  PATH_TO_DISCOVERIES_OR_CHECKPOINT_A \
  PATH_TO_DISCOVERIES_OR_CHECKPOINT_B \
  --config_file PATH_TO_ANALYSIS_CONFIG \
  --label IMGEP \
  --label baseline
```

Pass one or more directories and repeat `--label` in the same order when custom
labels are needed. Discovery and checkpoint inputs can be mixed in one run.

The CLI writes a new run directory under `analysis_runs/` in the current
working directory by default. Use `--output_dir` to choose a different
destination. Analysis runs started from the visualization UI are written under
`<save_location>/analysis_runs/`.

When a broad parent directory is selected by mistake, nested
`analysis_runs/` directories are excluded from dataset discovery. Random
baseline discoveries remain usable by selecting their individual
`random_run_*/discoveries` directory as an analysis input.

## Run Analysis From The UI

The `Analysis` page of the visualization server provides two actions:

- `Generate Discoveries`
  - runs a random baseline from a config file,
  - writes discoveries that can later be analyzed.
- `Analyze Discoveries`
  - runs the offline analysis stack across all discovery or checkpoint paths entered in the dataset list,
  - uses the analysis config file entered in the page,
  - renders module images from the generated analysis summary.

Minimal launch flow:

```bash
python -m adtool.user_tools.visu.server \
  --discoveries PATH_TO_DISCOVERIES \
  --config_file PATH_TO_ANALYSIS_CONFIG
```

Then:

1. Open the `Analysis` page.
2. Enter the analysis config file.
3. Add one or more discovery or checkpoint directories.
4. Click `Run analysis`.

## Design Rules

Use the analysis layer when:

- the metric needs all discoveries,
- the metric is naturally offline,
- the output is a plot, image, progression, comparison, or batch summary.

Do not use the analysis layer for:

- live exploration control,
- per-point interactive filtering in the discovery map,
- behavior that must run during the experiment loop.

Those belong to the visualization layer or to the exploration system itself.

## Related Docs

- [Visualization Guide](./VISUALIZATION.md)
- [Visualization UI Guide](./VISUAL_UI_GUIDE.md)
