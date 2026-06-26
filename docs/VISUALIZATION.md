# Visualization Guide

## What This Layer Is

The visualization module is the interactive web interface for exploring saved discoveries and, optionally, interacting with a live experiment.

It is separate from the exploration pipeline itself:

- the experiment writes discoveries,
- the visualization server reads those discoveries,
- optional system-specific helpers can enrich the UI with highlights, live controls, and goal targeting.

The visualization layer is useful for:

- browsing discoveries in a 2D map,
- switching projection methods,
- selecting and exporting discoveries,
- running offline analysis from the UI,
- refreshing the view while an experiment is still running,
- pausing and resuming a live experiment,
- highlighting discoveries from system-defined rules,
- steering a compatible goal sampler toward specific 2D regions.

## Start The Viewer

Manual mode:

```bash
python -m adtool.examples.visu.server \
  --discoveries PATH_TO_DISCOVERIES
```

With analysis-side extensions such as highlights:

```bash
python -m adtool.examples.visu.server \
  --discoveries PATH_TO_DISCOVERIES \
  --config_file PATH_TO_ANALYSIS_CONFIG
```

With live online updates:

```bash
python -m adtool.examples.visu.server \
  --discoveries PATH_TO_DISCOVERIES \
  --config_file PATH_TO_ANALYSIS_CONFIG \
  --refresh
```

`--config_file` is optional, but it becomes important when you want the viewer to load extra system-specific features such as `discovery_highlights`, or when refresh mode should expose goal zones for a goal-oriented sampler.

## What The Viewer Consumes

The viewer primarily consumes:

- a discoveries directory,
- rendered media already produced by the experiment,
- optional saved `filters` inside each `discovery.json`,
- optional visualization config passed through `--config_file`.

When `--config_file` is provided, the viewer can also consume top-level visualization hooks such as:

```json
{
  "goal_oriented_sampler": true,
  "discovery_highlights": {
    "path": "adtool.examples.my_system.helpers.discovery_highlights.MyHighlights",
    "config": {}
  }
}
```

## Offline And Online Viewer Behavior

The viewer has two runtime modes.

### Manual mode

Manual mode is the default when `--refresh` is not passed.

In this mode:

- discoveries are loaded from disk when the viewer starts,
- `Refresh` reloads the current discovery export,
- `Recompute Layout` recomputes the 2D projection,
- there is no pause/resume control,
- goal zones are unavailable.

### Refresh mode

Refresh mode is enabled with `--refresh`.

In this mode:

- the viewer watches the discoveries folder,
- new discoveries are added to the displayed set periodically,
- missing highlight filters are computed automatically for new discoveries when a highlight provider is available,
- the current point cloud is incrementally refreshed more often than the full projection layout.

Current behavior:

- point updates happen every few seconds,
- full layout recomputation happens much less often,
- pause/resume becomes available,
- goal zones may become available if the config and projection support them.

### What pause does

Pause is only available in refresh mode.

The pause button writes the experiment control state used by the running experiment. In practice, it tells a compatible experiment loop to temporarily stop advancing new discoveries until resumed.

Pause does not:

- close the viewer,
- delete discoveries,
- stop offline analysis jobs that are already running.

Pause only affects live experimentation that is reading the same experiment-control state.

## Discovery Highlights

### What highlights are

Highlights are a visualization-side rule system that colors or hides points according to system-defined per-discovery values.

Typical examples:

- highlight discoveries with large L2 miss counts,
- hide discoveries outside a desired timing range,
- mark discoveries whose program structure satisfies a condition.

Highlighting is not part of the offline analysis-module system. It belongs to the viewer.

### Why filter computation is user-triggered

Highlight rules operate on saved `filters` stored in each discovery file.

These filters are not precomputed automatically in manual mode. They are materialized when you explicitly click `Compute Filters`.

In refresh mode, when a highlight provider is available, newly discovered points get their missing filters computed automatically during online updates.

This keeps the base experiment output minimal and avoids paying the filter-computation cost unless the user wants highlightable values.

### Highlight config hook

The provider is declared in the top-level `discovery_highlights` section of the analysis/visualization config:

```json
{
  "discovery_highlights": {
    "path": "adtool.examples.my_system.helpers.discovery_highlights.MyHighlights",
    "config": {}
  }
}
```

### Implement a highlight provider

Highlight providers inherit `DiscoveryHighlightProvider`.

They must define:

- `fields()`
  - the selectable dimensions shown in the UI,
- `rules()`
  - the default predefined rules initially offered to the user,
- `compute_filters(discovery_payload)`
  - how to extract flat per-discovery values from `discovery.json`.

Minimal template:

```python
from adtool.examples.visu.highlights import (
    DiscoveryHighlightField,
    DiscoveryHighlightProvider,
    DiscoveryHighlightRule,
)


class MyHighlights(DiscoveryHighlightProvider):
    def fields(self):
        return [
            DiscoveryHighlightField(
                field_id="score",
                label="Score",
                value_type="number",
                min=0,
                max=100,
            ),
        ]

    def rules(self):
        return [
            DiscoveryHighlightRule(
                rule_id="score_high",
                label="High score",
                field_id="score",
                clauses=[{"lower": 70, "upper": 100}],
            ),
        ]

    def compute_filters(self, discovery_payload):
        return {
            "score": float(discovery_payload["raw_output"]["score"]),
        }
```

### What the viewer does with filters

When a provider exists:

- the viewer exports a highlight schema,
- the `Compute Filters` action asks the backend to materialize `filters` into discovery files,
- the viewer reads those filters and evaluates highlight rules client-side,
- points can be shown normally, highlighted, or hidden depending on rule mode.

If no provider is configured:

- the highlight panel is hidden,
- no filter materialization action is exposed.

## Goal Zones

### What goal zones are

Goal zones are a live steering feature for compatible goal samplers.

You place a 2D zone on the current projection, and the sampler is asked to generate goals inside that zone more often while still keeping some exploration outside it.

It is possible to place multiple zones. They behave as one combined target region: if a point falls inside any configured zone, it is considered inside the goal-targeted area.

### When the feature is available

Goal zones are available only when all of these conditions are true:

- the viewer runs with `--refresh`,
- the config file passed to the viewer contains `"goal_oriented_sampler": true`,
- the current projection method supports zone resolution,
- the runtime layout is already ready enough to resolve the zone.

Currently supported projection methods:

- `axis`
- `pca`

When these conditions are not met, the feature is hidden or disabled with an explanatory message in the UI.

### How the viewer writes goal targeting

The viewer stores goal-targeting state in experiment control. It resolves the 2D UI zone into a sampler-facing payload under `goal_targeting`.

The exact resolved payload depends on the projection:

- `axis`
  - stores projection axes plus normalization info,
- `pca`
  - stores PCA components and normalization info.

### How a sampler consumes goal targeting

A compatible sampler must accept an optional `goal_targeting` argument and use it when sampling goals.

Minimal shape:

```python
class MyZoneGoalSampler:
    def sample(
        self,
        history,
        feature_size,
        min_=None,
        max_=None,
        goal_targeting=None,
        **kwargs,
    ):
        if goal_targeting is None or not goal_targeting.get("zones"):
            return self.sample_default(history, feature_size, min_=min_, max_=max_)

        # Sample more often inside the requested 2D zone.
        return self.sample_targeted(history, feature_size, goal_targeting, min_=min_, max_=max_)
```

For a concrete reference, see:

- [interference_zone_goal_sampler.py](../examples/embedded_systems/examples/core_interferences/behavior_map/goal_sampler/interference_zone_goal_sampler.py)

## Analysis From The Viewer

The `Analysis` page can launch:

- a random baseline generation run,
- an offline analysis run against one or more comparison discovery folders.

The viewer does not implement analysis logic itself. It forwards the request to the same offline analysis runner described in [Analysis Modules](./ANALYSIS_MODULES.md).

## Related Docs

- [Analysis Modules](./ANALYSIS_MODULES.md)
- [Visualization UI Guide](./VISUAL_UI_GUIDE.md)
