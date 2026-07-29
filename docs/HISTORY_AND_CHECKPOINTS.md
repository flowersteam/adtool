# History and checkpoints

`HistoryStore` is the explorer's single interface for discovery history. It
keeps the data needed to continue an experiment separate from the optional
files used to inspect or visualize discoveries. The current implementation
uses local pickle and JSON files; it does not use a database.

## Logging

Set `experiment.config.log_level` to `DEBUG`, `INFO`, `WARNING`, `ERROR`, or
`CRITICAL` (case-insensitive). It defaults to `INFO`.

Use `DEBUG` when inspecting history and checkpoints. It logs cache updates,
retrieval plans, loaded checkpoint chunks, nearest-neighbour selections, and
checkpoint batches in a compact `key=value` format. `INFO` keeps only normal
experiment progress and checkpoint messages.

## History configuration defaults

| Setting | Default | Meaning |
| --- | --- | --- |
| `save_frequency` | Required in JSON config | Number of discoveries between checkpoints. `ExperimentPipeline` uses `100` only when constructed directly in Python without this argument. |
| `discoveries_cache_size` | `save_frequency` | Number of newest discoveries retained in RAM. `0` disables the cache; `-1` retains complete history. |
| `history_lookback_length` | `-1` | Number of newest discoveries available to an explorer. `-1` means all available history. |

## What is saved where?

`save_location` contains two independent kinds of output:

- `checkpoints/` contains the complete experiment state. Use these folders to
  resume or branch an experiment.
- `discoveries/` contains `discovery.json` files and optional renders. These
  are for visualization and analysis only; a normal experiment run never
  reloads them as history.

```text
<save_location>/
├── discoveries/
│   ├── config.json
│   └── <timestamp>_exp_<id>_idx_<step>_seed_<seed>/
│       ├── discovery.json
│       └── visu.<extension>              # when rendering is enabled
└── checkpoints/
    └── step-00000042-cfg-<config-hash>-branch-<branch-id>/
        ├── manifest.json
        ├── pipeline.pickle
        ├── pipeline__explorer.pickle
        ├── ... one pickle for each Leaf component
        └── history-step-00000042-cfg-<config-hash>-branch-<branch-id>.pickle
```

The history file is present only when the checkpoint owns new discoveries.
Its descriptive name identifies the step and branch even when viewed outside
its checkpoint folder.

## Checkpoints and chunks

`save_frequency` controls when checkpoints are written. A checkpoint is made
after every `save_frequency` discoveries and once more at the end of a
non-empty requested run if there are unsaved discoveries.

A **history chunk** is the batch of discoveries made since the checkpoint's
parent. Normally it contains at most `save_frequency` discoveries. It is not
a fixed-size cache and it is not the complete experiment history.

For example, with `save_frequency: 5`, checkpoints at steps 5 and 10 contain
two separate five-discovery chunks. The step-10 checkpoint points to step 5;
it does not copy the first five discoveries again.

```text
step 5  ── owns discoveries 1–5
   ↑
step 10 ── owns discoveries 6–10
   ↑
step 15 ── owns discoveries 11–15
```

The manifest records the component files, history files and their discovery
counts, plus `parent_checkpoint`. On resume, the history store reads this
parent chain once and keeps the small list of chunk references in RAM. It
loads a chunk file only when a retrieval needs discoveries that are not in the
RAM cache.

Saving is safe against an interrupted write: files are first written to a
temporary checkpoint directory, then that complete directory is atomically
renamed into place. There is no separate staging directory.

**Keep every parent checkpoint and its files while any child checkpoint may be
resumed.** A child branch depends on its parent chain for older history.

## RAM discovery cache

`experiment.config.discoveries_cache_size` controls a rolling cache of the
newest decoded discoveries. It is independent from the pending batch that will
be written at the next checkpoint.

| Value | Behaviour |
| --- | --- |
| Positive integer | Keep that many newest discoveries in RAM. Requests contained in this window do not read history pickles. |
| `0` | Disable the cache. Required completed chunks are read from disk. The pipeline emits a warning. |
| `-1` | Keep complete history in RAM. Retrieval does not read checkpoint history files after initialization, but memory grows with every discovery. The pipeline emits a warning. |

When a checkpoint is resumed, the cache is rebuilt from the newest saved
chunks. Complete-cache mode (`-1`) therefore reads the saved history once at
resume time; later retrievals use RAM.

`experiment.config.history_lookback_length` limits an explorer to the newest N
discoveries. `-1` means all available history. If the requested window is
larger than the RAM cache, older chunks are streamed in chronological order;
the whole history is not retained in memory merely to perform the search.

## How explorers use history

Explorers use `HistoryStore`, rather than accessing files or another explorer's
memory directly:

- `record()` adds a discovery to the pending checkpoint batch and RAM cache.
- `last()` returns the latest discovery.
- `nearest()` performs an exact chunk-by-chunk nearest-neighbour search; there
  is currently no nearest-neighbour index.
- `random()` selects a valid history item without materializing all records.
- `feature_bounds()` calculates numeric bounds in one streaming pass.

For normalized nearest-neighbour selection, a caller may pass bounds already
calculated for goal sampling. This avoids repeating the bounds pass. The
nearest-neighbour search itself remains a separate streaming pass.

`iter_history()` and `iter_chunks()` are retained for custom export or
inspection code. `features()` builds a full feature matrix and should only be
used by external algorithms that explicitly require one.

## Fresh runs, resume, and branches

Without a resume setting, an experiment starts with a fresh system, explorer,
and empty history. Existing `discoveries/*/discovery.json` files are ignored.

To resume, set `resume_checkpoint` to a folder name directly under
`<save_location>/checkpoints/`:

```json
{
  "experiment": {
    "config": {
      "save_location": "./runs/grayscott",
      "save_frequency": 10,
      "log_level": "INFO",
      "discoveries_cache_size": 500,
      "history_lookback_length": 500,
      "bootstrap_size": 1,
      "render_every": 0,
      "resume_checkpoint": "step-00000030-cfg-a1b2c3d4e5f6-branch-1234abcd"
    }
  }
}
```

An absolute checkpoint path, or a path relative to `save_location`, also
works.
Resume restores the saved component state, discovery history, and step. The
continued run receives a new branch ID. Its first new checkpoint points back
to the selected checkpoint, so the original experiment is preserved and the
new discoveries form a branch rather than overwriting it.

## Typical workflow

1. Choose a local `save_location` and a positive `save_frequency`.
2. Optionally set `discoveries_cache_size` and `history_lookback_length`.
3. Run:

   ```bash
   python -m adtool.runners.run_experimentations \
     --config_file config.json --nb_iterations 30
   ```

4. Find checkpoints:

   ```bash
   find <save_location>/checkpoints -name manifest.json -exec dirname {} \;
   ```

5. Put the chosen folder name in `resume_checkpoint` and run again.

`CheckpointStore` is the persistence boundary and `FileCheckpointStore` is
the current local-files implementation. A future storage backend can replace
it without changing explorer history access.
