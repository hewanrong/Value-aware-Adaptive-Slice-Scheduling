# Experiment Protocol

This project separates formal code, documentation, and run artifacts so future experiments stay reproducible and the code context remains small.

## Repository Boundaries

- `src/` contains formal library code.
- `scripts/` contains formal command-line experiment entrypoints.
- `docs/` contains phase documents, protocol notes, and frozen experiment summaries.
- `configs/` contains reusable experiment configurations. Use `configs/panda_3840.yaml` for default PANDA experiments unless a task explicitly asks for raw resolution.
- `cache/` contains generated intermediate files and should not be treated as code context.
- `results/` contains generated outputs, figures, metrics, and sanity CSVs and should not be modified as source code.
- `sandbox/` or `experiments/`, if created later, are only for temporary exploration and should not become required runtime paths.

## Phase Discipline

- Phase 0 is frozen as dataset and slice sanity checking.
- Phase 1 may add GT mapping, real visual features, and real JPEG cost profiling.
- ViTDet integration and model training remain out of scope until explicitly requested in a later phase.

## Codex Task Defaults

For future Codex tasks, assume:

- Use `configs/panda_3840.yaml` unless the user explicitly requests `configs/panda_raw.yaml`.
- Do not rely on generated files in `cache/` or `results/` as source of truth unless the task is specifically about inspecting run artifacts.
- Do not modify scheduler or simulator behavior when the task is limited to dataset, slicing, or feature sanity checks.
