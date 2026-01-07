# Repository Guidelines

## Project Structure & Module Organization
Treat `implementation_tasks_checklist.md` as the source of truth for how the codebase should grow. Python orchestration code belongs under `python/` with `python/main.py` as the CLI entry, supporting packages like `python/pipeline/`, `python/instrumentation/`, and `python/utils/`. Native acceleration lives in `ScoreCalculator/` (C++/CUDA) and any shared helpers (e.g., `Utilities/`). Configuration, logs, and outputs should reside in `configs/`, `logs/`, and `output/` respectively so the instrumentation hooks referenced in the checklist can emit consistent artifacts such as `logs/timing.jsonl` and `output/result.json`.

## Build, Test, and Development Commands
Always work inside the `new_env_testing` environment before touching Python entry points:
```bash
pyenv activate new_env_testing
python -m python.main --config configs/default.json
```
Generate native bindings via CMake (see `reference_CMakeLists.txt` for dependency toggles). CUDA is currently disabled, so keep the flag off unless explicitly re-enabled:
```bash
cmake -S . -B build -DENABLE_CUDA_SUPPORT=OFF
cmake --build build -j$(nproc)
```
Run C++ unit tests through CTest / GoogleTest: `ctest --test-dir build`. Python invariants rely on pytest: `pytest tests -k invariants`. Benchmarks should live under `bench/` and emit comparable records by reusing `logs/timing.jsonl`.

## Coding Style & Naming Conventions
C++ targets C++17 with optional CUDA; keep headers self-contained, prefer `Eigen::` and `std::` namespaces explicitly, and follow 4-space indentation with brace-on-same-line (per the reference CMake flags). Python modules follow PEP 8 plus type-annotated functions. Name instrumentation constants exactly as listed in the checklist (e.g., `python/preprocess_total`). Filenames stay snake_case for Python, PascalCase for C++ classes, and `S_*` for aggregated score tensors.

## Testing Guidelines
Implement pytest suites under `tests/` mirroring pipeline phases (e.g., `test_invariants.py`). For C++, colocate GoogleTest targets inside `Tests/` and register them so `ctest` runs automatically. Tests should assert determinism: same config + seed ⇒ identical `Pfin`. Use JSONL instrumentation to assert coverage of each `*_total` span.

## Commit & Pull Request Guidelines
No Git history exists yet, so adopt the Issue naming guidance in `implementation_tasks_checklist.md`: prefix commits/PRs with the phase marker (`[P1]`, `[C]`, etc.) followed by an imperative summary ("[P2] Add GeoFilter bindings"). PRs must describe the affected pipeline sections, mention configs or datasets touched, reference instrumentation you added, and attach logs or screenshots demonstrating `python/main.py` and `ctest`/`pytest` results.

## Security & Configuration Tips
Pin local toolchains to the versions declared in `reference_CMakeLists.txt` (VTK 9.2, Eigen 3.3, Pybind11; CUDA linkage remains off until further notice). Never run Python entry points without `pyenv activate new_env_testing`; doing so can mis-link OpenMP and invalidate logged metrics. Store sensitive calibration data outside the repo and reference it via environment variables in configs.

### Handling Local Modifications
Before running commands that may overwrite files (e.g., regenerating bindings, re-running pipelines that write to configs or outputs), the developer or automation agent should:
- Use Git to check for local changes to tracked files, for example:
  - `git status` to see which files are modified.
  - `git diff` to inspect what changed.
- Present a summary of the changes to the user and explicitly ask whether to keep the modifications or discard them.
  - If the user wants to keep the changes, ensure they are committed (or stashed) before proceeding.
  - If the user wants to discard the changes, commit them and add a tag, then revert them with Git (for example, `git restore <file>` or `git reset --hard` as appropriate) before continuing.
