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

Analysis is driven by a config file with a top-level `analysis_modules` list:

```json
{
  "analysis_modules": [
    {
      "path": "adtool.user_tools.analysis_metrics.comparison_1d.Comparison1DModule",
      "config": {}
    },
    {
      "path": "examples.program_based_systems.examples.core_interferences.analysis_modules.MutualMissHeatmapModule",
      "config": {}
    }
  ]
}
```

Each entry is loaded independently, in order. The module order in this list is also the display order on the analysis page.

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
          "alpha": 0.35,
          "figsize": [7.0, 4.0]
        }
      }
    }
  ]
}
```

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
          "path": "examples.program_based_systems.examples.core_interferences.behavior_map.space_coverage.GridSpaceCoverageMetric",
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
  PATH_TO_PRIMARY_DISCOVERIES \
  PATH_TO_COMPARISON_DISCOVERIES \
  --config_file PATH_TO_ANALYSIS_CONFIG \
  --primary_label IMGEP \
  --comparison_label baseline
```

To compare against multiple datasets, pass multiple discovery directories and repeat `--comparison_label` as needed.

The command writes a new run directory under `analysis_runs/` by default.

## Run Analysis From The UI

The `Analysis` page of the visualization server provides two actions:

- `Generate Discoveries`
  - runs a random baseline from a config file,
  - writes discoveries that can later be analyzed.
- `Analyze Discoveries`
  - runs the offline analysis stack on the current discoveries folder against one or more comparison folders,
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
3. Add one or more comparison discovery folders.
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
