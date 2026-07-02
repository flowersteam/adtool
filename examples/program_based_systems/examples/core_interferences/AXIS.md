# Core Interference Visualization Axes

This note documents what the visualization axes mean for the
`core_interference` example when discoveries are used as visualization input.

## When axis ids are meaningful

Axis ids only have a direct metric meaning when the visualization projection
method is set to `axis`.

- `axis 0` means `discovery.json["output"][0]`
- `axis 1` means `discovery.json["output"][1]`
- more generally, `axis N` means `discovery.json["output"][N]`

If the projection method is `umap`, `pca`, or `tsne`, the displayed X/Y axes are
synthetic 2D coordinates built from the whole output vector. In that case, the
screen axes do not correspond to named raw metrics.

The visualizer defaults to `umap`, so the metric mapping below matters only
after switching the projection method to `axis`.

## Where the output vector comes from

For this example, the `output` vector is produced by
`behavior_map/encoder/interference_metric_encoder.py`.

The encoder keeps selected metrics from `raw_output["mutual"]` and flattens each
selected array with `reshape(-1)`. In practice, the axis order therefore follows
the insertion order of the `mutual` dictionary created in
`systems/runner/interference_env_simulator_runner.py`.

## Default `core_interference.json` mapping

For the default config in `core_interference.json`:

- `num_banks = 4`
- `num_addr = 48`

the emitted matrix metrics have shape `4 x 4` in the saved output vector:

- 4 rows because `Experiment` uses `num_rows = num_addr // 16 + 1`
- 4 banks because `num_banks = 4`

That gives the following axis layout.

### Scalar axes

- `axis 0`: `diff_time_core0`
- `axis 1`: `diff_time_core1`
- `axis 2`: `diff_time`

### Matrix blocks

The next four metric groups are flattened in row-major order:

- `axes 3..18`: `miss_core0`
- `axes 19..34`: `miss_core1`
- `axes 35..50`: `hits_core0`
- `axes 51..66`: `hits_core1`

Inside each `4 x 4` block:

- local index `0` is `[row 0, bank 0]`
- local index `1` is `[row 0, bank 1]`
- local index `2` is `[row 0, bank 2]`
- local index `3` is `[row 0, bank 3]`
- local index `4` is `[row 1, bank 0]`
- ...
- local index `15` is `[row 3, bank 3]`

Examples:

- `axis 3` is `miss_core0[0, 0]`
- `axis 18` is `miss_core0[3, 3]`
- `axis 19` is `miss_core1[0, 0]`
- `axis 35` is `hits_core0[0, 0]`
- `axis 66` is `hits_core1[3, 3]`

### L2 scalar axes

- `axis 67`: `L2_miss_read_core0`
- `axis 68`: `L2_hit_read_core0`
- `axis 69`: `L2_miss_read_core1`
- `axis 70`: `L2_hit_read_core1`
- `axis 71`: `L2_miss_write_core0`
- `axis 72`: `L2_hit_write_core0`
- `axis 73`: `L2_miss_write_core1`
- `axis 74`: `L2_hit_write_core1`

## What changes if the config changes

This mapping is not fully stable across config changes.

- If `num_banks` changes, the size of each matrix block changes.
- If `num_addr` changes, the number of rows changes because the simulator code
  derives rows from `num_addr // 16 + 1`.
- If the encoder selection or the `mutual` payload construction order changes,
  axis ids will shift.

So the axis table above is accurate for the current default
`examples/program_based_systems/examples/core_interferences/core_interference.json`
example, not as a universal guarantee for every future variant.