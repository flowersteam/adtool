# Legacy Cleanup TODO

This file tracks legacy and dead-code cleanup identified in `adtool/`, the shipped example configs, and the surrounding docs/config paths.

Notes from the audit:
- `adtool.explorers.CuriosityIMGEPExplorer.IMGEPExplorer` is still referenced by `examples/flashlenia/flashlenia.json` and `tests/flashlenia/smoke_config.json`. It is not dead code in the current checkout.
- `adtool.wrappers.BoxProjector` is still used by many shipped example statistics/behavior maps. It is legacy in naming/location, but not removable yet.
- Goal targeting already has a newer control path through `experiment_control.json`; the old `discoveries/target.json` path is the legacy part.
- A large part of the current persistence stack is dead or deprecated but still coupled into normal imports through `Leaf`/locator code.

## Core runtime and checkpoint legacy
- [ ] Remove `experiment.config.legacy_checkpoint_saves` support from [`adtool/ExperimentPipeline.py`](/home/arthur/Documents/INRIA/codes/adtool/adtool/ExperimentPipeline.py) and stop branching into recursive object checkpoint saves.
- [ ] Remove `experiment.config.resume_from_uid` support from [`adtool/runners/run_experimentations.py`](/home/arthur/Documents/INRIA/codes/adtool/adtool/runners/run_experimentations.py).
- [ ] Remove legacy checkpoint-specific `save()` behavior in [`adtool/ExperimentPipeline.py`](/home/arthur/Documents/INRIA/codes/adtool/adtool/ExperimentPipeline.py) so saving only means discovery persistence.
- [ ] Remove the duplicate legacy pipeline in [`adtool/ExperimentPipelineVariance.py`](/home/arthur/Documents/INRIA/codes/adtool/adtool/ExperimentPipelineVariance.py).
- [ ] Remove the stale commented import of `ExperimentPipelineVariance` from [`adtool/runners/run_experimentations.py`](/home/arthur/Documents/INRIA/codes/adtool/adtool/runners/run_experimentations.py).
- [ ] Remove the old `discoveries/target.json` fetch from [`adtool/ExperimentPipeline.py`](/home/arthur/Documents/INRIA/codes/adtool/adtool/ExperimentPipeline.py).
- [ ] Remove the old `discoveries/target.json` fetch from [`adtool/ExperimentPipelineVariance.py`](/home/arthur/Documents/INRIA/codes/adtool/adtool/ExperimentPipelineVariance.py).
- [ ] Keep goal targeting driven only by [`adtool/utils/interaction/experiment_control.py`](/home/arthur/Documents/INRIA/codes/adtool/adtool/utils/interaction/experiment_control.py) and the resolved `goal_targeting` payload.

## Leaf persistence and locator dead code
- [ ] Remove recursive object-state persistence from [`adtool/utils/leaf/Leaf.py`](/home/arthur/Documents/INRIA/codes/adtool/adtool/utils/leaf/Leaf.py): `save_leaf`, `load_leaf`, `_pointerize_submodules`, `_load_leaf_submodules_recursively`, `_retrieve_parent_locator_class`, and `_get_uid_base_case`.
- [ ] Remove persistence-only helper types and imports from [`adtool/utils/leaf/Leaf.py`](/home/arthur/Documents/INRIA/codes/adtool/adtool/utils/leaf/Leaf.py) once checkpoint persistence is deleted.
- [ ] Remove [`adtool/utils/leaf/LeafUID.py`](/home/arthur/Documents/INRIA/codes/adtool/adtool/utils/leaf/LeafUID.py) if nothing remains that needs a persistent leaf identifier.
- [ ] Remove legacy locator implementations that only exist for leaf persistence:
  - [`adtool/utils/leaf/locators/Locator.py`](/home/arthur/Documents/INRIA/codes/adtool/adtool/utils/leaf/locators/Locator.py)
  - [`adtool/utils/leaf/locators/LinearBase.py`](/home/arthur/Documents/INRIA/codes/adtool/adtool/utils/leaf/locators/LinearBase.py)
  - [`adtool/utils/leaf/locators/locators.py`](/home/arthur/Documents/INRIA/codes/adtool/adtool/utils/leaf/locators/locators.py)
  - [`adtool/utils/leafutils/leafintegrations/expedb_locators.py`](/home/arthur/Documents/INRIA/codes/adtool/adtool/utils/leafutils/leafintegrations/expedb_locators.py)
- [ ] Remove the import-time coupling where `BlobLocator` currently pulls in the linear/SQLAlchemy locator stack even when only active runtime code is imported.
- [ ] Remove any now-unused `BlobLocator` assignments across core modules after the checkpoint/persistence path is eliminated.

## Wrapper cleanup
- [ ] Remove [`adtool/wrappers/SaveWrapper.py`](/home/arthur/Documents/INRIA/codes/adtool/adtool/wrappers/SaveWrapper.py).
- [ ] Remove [`adtool/wrappers/TransformWrapper.py`](/home/arthur/Documents/INRIA/codes/adtool/adtool/wrappers/TransformWrapper.py).
- [ ] Remove [`adtool/wrappers/IdentityWrapper.py`](/home/arthur/Documents/INRIA/codes/adtool/adtool/wrappers/IdentityWrapper.py).
- [ ] Remove [`adtool/wrappers/WrapperPipeline.py`](/home/arthur/Documents/INRIA/codes/adtool/adtool/wrappers/WrapperPipeline.py).
- [ ] Clean up [`adtool/wrappers/__init__.py`](/home/arthur/Documents/INRIA/codes/adtool/adtool/wrappers/__init__.py) so it stops exporting deprecated wrapper classes and stale wrapper docs.
- [ ] Replace `IdentityWrapper()` defaults in [`adtool/explorers/IMGEPExplorer.py`](/home/arthur/Documents/INRIA/codes/adtool/adtool/explorers/IMGEPExplorer.py) with non-wrapper defaults that match the post-wrapper architecture.
- [ ] Replace `IdentityWrapper()` defaults in [`adtool/explorers/CuriosityIMGEPExplorer.py`](/home/arthur/Documents/INRIA/codes/adtool/adtool/explorers/CuriosityIMGEPExplorer.py) with non-wrapper defaults that match the post-wrapper architecture.
- [ ] Remove commented `SaveWrapper` remnants from [`adtool/maps/MeanBehaviorMap.py`](/home/arthur/Documents/INRIA/codes/adtool/adtool/maps/MeanBehaviorMap.py) and [`adtool/maps/UniformParameterMap.py`](/home/arthur/Documents/INRIA/codes/adtool/adtool/maps/UniformParameterMap.py).
- [ ] Re-check [`adtool/explorers/history_store.py`](/home/arthur/Documents/INRIA/codes/adtool/adtool/explorers/history_store.py) after `SaveWrapper` removal and delete the compatibility wording/API surface that only exists for old `SaveWrapper.map(...)` call sites.

## CPPN and NEAT code that should leave the main library
- [ ] Move [`adtool/wrappers/CPPNWrapper.py`](/home/arthur/Documents/INRIA/codes/adtool/adtool/wrappers/CPPNWrapper.py) into the examples that actually use CPPN-based initialization.
- [ ] Move [`adtool/maps/NEATParameterMap.py`](/home/arthur/Documents/INRIA/codes/adtool/adtool/maps/NEATParameterMap.py) into the examples that actually use it.
- [ ] Remove [`adtool/maps/cppn/`](/home/arthur/Documents/INRIA/codes/adtool/adtool/maps/cppn) from the main library after the example-local replacements exist.
- [ ] Update Lenia example imports and config paths to stop referencing `adtool.wrappers.CPPNWrapper` and `adtool.maps.NEATParameterMap`.
- [ ] Update FlowLenia example imports and config paths to stop referencing `adtool.wrappers.CPPNWrapper` and `adtool.maps.NEATParameterMap`.
- [ ] Re-check example-local code after the move so only CPPN examples carry the NEAT/CPPN dependency chain.

## Explorer and map dead code
- [ ] Remove [`adtool/explorers/IMGEPExplorerInterpolation.py`](/home/arthur/Documents/INRIA/codes/adtool/adtool/explorers/IMGEPExplorerInterpolation.py).
- [ ] Clean up [`adtool/explorers/__init__.py`](/home/arthur/Documents/INRIA/codes/adtool/adtool/explorers/__init__.py), which already carries stale commented exports.
- [ ] Remove any references to interpolation-based explorer behavior from docs or comments once the file is deleted.
- [ ] Rename or relocate legacy-named core maps so the remaining API stops pretending every component is a "map" when some are samplers/projectors/helpers.
- [ ] Revisit [`adtool/maps/IdentityBehaviorMap.py`](/home/arthur/Documents/INRIA/codes/adtool/adtool/maps/IdentityBehaviorMap.py) once wrapper cleanup is done; it appears to survive mainly for helper/demo usage rather than the shipped JSON configs.
- [ ] Revisit [`adtool/maps/MeanBehaviorMap.py`](/home/arthur/Documents/INRIA/codes/adtool/adtool/maps/MeanBehaviorMap.py) once wrapper cleanup is done; it is still the default explorer config path but not directly referenced by shipped example/test JSON.

## Callback dead code
- [ ] Remove [`adtool/callbacks/custom_print_callback.py`](/home/arthur/Documents/INRIA/codes/adtool/adtool/callbacks/custom_print_callback.py).
- [ ] Remove [`adtool/callbacks/on_discovery_callbacks/on_discovery_save_callback_on_disk.py`](/home/arthur/Documents/INRIA/codes/adtool/adtool/callbacks/on_discovery_callbacks/on_discovery_save_callback_on_disk.py).
- [ ] Remove [`adtool/callbacks/on_discovery_callbacks/save_discovery_in_expedb.py`](/home/arthur/Documents/INRIA/codes/adtool/adtool/callbacks/on_discovery_callbacks/save_discovery_in_expedb.py).
- [ ] Remove [`adtool/callbacks/on_save_callbacks/on_save_modules_on_disk_callback.py`](/home/arthur/Documents/INRIA/codes/adtool/adtool/callbacks/on_save_callbacks/on_save_modules_on_disk_callback.py).
- [ ] Remove [`adtool/callbacks/on_save_finished_callbacks/generate_report_callback.py`](/home/arthur/Documents/INRIA/codes/adtool/adtool/callbacks/on_save_finished_callbacks/generate_report_callback.py).
- [ ] Clean up `__init__.py` exports that still re-export unused callback types:
  - [`adtool/callbacks/__init__.py`](/home/arthur/Documents/INRIA/codes/adtool/adtool/callbacks/__init__.py)
  - [`adtool/callbacks/on_discovery_callbacks/__init__.py`](/home/arthur/Documents/INRIA/codes/adtool/adtool/callbacks/on_discovery_callbacks/__init__.py)
  - [`adtool/callbacks/on_save_callbacks/__init__.py`](/home/arthur/Documents/INRIA/codes/adtool/adtool/callbacks/on_save_callbacks/__init__.py)
- [ ] Remove event callback plumbing from [`adtool/ExperimentPipeline.py`](/home/arthur/Documents/INRIA/codes/adtool/adtool/ExperimentPipeline.py) that only exists for deleted callback families if those hooks become unreachable.
- [ ] Keep [`adtool/callbacks/on_discovery_callbacks/save_discovery_on_disk.py`](/home/arthur/Documents/INRIA/codes/adtool/adtool/callbacks/on_discovery_callbacks/save_discovery_on_disk.py) as the currently proven callback path used by shipped examples/tests.

## Example and config fallout
- [ ] Update shipped JSON configs after CPPN/NEAT moves so every `path` still resolves.
- [ ] Update shipped JSON configs after callback cleanup so only supported callback classes remain.
- [ ] Remove stale JSON/config keys tied to deleted checkpoint behavior.
- [ ] Re-check the FlashLenia example and smoke test after cleanup so `CuriosityIMGEPExplorer` is preserved as an actively used explorer, not removed as dead code.
- [ ] Re-check whether [`examples/draft/`](/home/arthur/Documents/INRIA/codes/adtool/examples/draft) should remain as an intentionally empty scaffold or be dropped as dead maintenance surface.

## Documentation and repo guidance cleanup
- [ ] Fix [`AGENTS.md`](/home/arthur/Documents/INRIA/codes/adtool/AGENTS.md) so it stops pointing to nonexistent `adtool/runtime/run.py`.
- [ ] Fix [`AGENTS.md`](/home/arthur/Documents/INRIA/codes/adtool/AGENTS.md) so it stops claiming explorer history depends on `SaveWrapper`.
- [ ] Update [`docs/execution_flow`](/home/arthur/Documents/INRIA/codes/adtool/docs/execution_flow) so it no longer documents `SaveWrapper` calls in the active execution path.
- [ ] Re-scan `README.md` and docs for stale mentions of deleted callbacks, checkpoint resume, wrapper-based history, and moved CPPN/NEAT code.

## Validation and cleanup follow-through
- [ ] Verify every shipped example JSON and smoke-test JSON still resolves after path updates.
- [ ] Verify replay from `discoveries/*/discovery.json` still works after all checkpoint persistence code is removed.
- [ ] Verify goal targeting still works through `experiment_control.json` with no `target.json` support.
- [ ] Verify dead dependency edges are gone from normal imports, especially the legacy locator/SQLAlchemy path.
- [ ] Remove any now-unused imports, comments, and compatibility shims left behind by the legacy cleanup.

## Needs discussion
- What to do about the database layer, and whether any of it should remain:
  - The old persistence path currently pulls in locator/database code even for normal imports.
  - Decide whether all file/SQLite/ExpeDB persistence code should be removed with legacy checkpoint saving, or whether a smaller database abstraction is still needed for some future non-legacy use.
  - This includes deciding the fate of `LinearBase`, locator factories, ExpeDB locators, and any SQLAlchemy dependency that only exists for the old save/reload path.
- What to do about leaf state saving, and whether it is still relevant at all:
  - The current runtime already resumes by replaying `discoveries/*/discovery.json`, which bypasses recursive object-state reload.
  - Decide whether `Leaf` should remain only as a lightweight composition/container utility, or whether its save/load/UID machinery has any real remaining use.
  - If leaf state save is no longer relevant, the whole object checkpoint model should disappear rather than stay as dormant compatibility code.
- What to do about [`adtool/utils/interaction/FeedbackQueueClient.py`](/home/arthur/Documents/INRIA/codes/adtool/adtool/utils/interaction/FeedbackQueueClient.py):
  - Decide whether it is part of any still-supported workflow or just leftover integration code.
  - If it is unused, it should be added to the removal work; if it is still needed, its supported entrypoints and ownership should be made explicit.
  - This also affects whether the rest of `adtool.utils.interaction` should be kept as-is or reduced to the experiment control path only.
- Whether all callbacks should really be removed:
  - Current shipped example/test JSON only prove active use of `SaveDiscoveryOnDisk`.
  - Decide whether the target state is “keep only discovery-saving callbacks”, “keep minimal event hook interfaces with no built-in legacy implementations”, or “remove the callback system entirely and hardwire the remaining save flow”.
  - This decision affects `ExperimentPipeline` hook structure, callback base classes, and config schema stability.
- Whether all maps and explorers that are not defaults should really be removed:
  - Some non-default modules are clearly dead, but some odd ones are still referenced by examples, such as `CuriosityIMGEPExplorer` in FlashLenia.
  - Decide whether the removal rule is “not default”, “not referenced by shipped configs/tests”, or “not part of the supported public surface”.
  - This also affects modules like `IdentityBehaviorMap`, `MeanBehaviorMap`, and any example-specific explorer/map that survives only through a single maintained example.
- The wrappers are now useless as an abstraction; decide how to handle the few still-relevant pieces:
  - `SaveWrapper`, `TransformWrapper`, `IdentityWrapper`, and `WrapperPipeline` look removable.
  - `BoxProjector` is still actively used, but the `wrapper` naming and location no longer match its role.
  - CPPN-related wrappers should move into the examples that use them.
  - Decide whether relevant survivors should keep the `wrapper` nomination for compatibility, or be migrated to clearer module names and locations now.
