# Holding Point Search Pipeline

Implements the Rev2 specification for robotic food-cutting grasp search. The system mixes Python orchestration (Open3D, trimesh) with Eigen/PCL-based C++ scoring exposed through pybind11 bindings.

## Environment Setup

1. Activate the required Python environment before running any entry point:

   ```bash
   pyenv activate new_env_testing
   ```

2. Install Python dependencies (Open3D, trimesh, numpy) according to your environment policy.

3. Build the native module via CMake; CUDA stays disabled per current requirements:

   ```bash
   cmake -S . -B build -DENABLE_CUDA_SUPPORT=OFF
   cmake --build build -j$(nproc)
   ```

## Running the Pipeline

Execute the integrated pipeline and write results to `output/result.json`:

```bash
python -m python.main --config configs/default.json --output output/result.json
```

The run performs: point-cloud preprocessing → valid index computation → geo filter (C++) → positional scores (C++) → contact-surface extraction + wrench (Python) → dynamics scores (C++) → Algorithm 1 accumulation. Timing JSONL is emitted under `logs/timing.jsonl` when instrumentation is enabled in the config.

## Benchmarking

Use the benchmark harness to sweep downsample sizes and force-sample counts. Each trial reuses the integrated pipeline and appends a JSON record to `logs/benchmark.jsonl`.

```bash
python -m bench.benchmark \
  --config configs/default.json \
  --downsample 64 128 256 \
  --force-samples 200 500 1000 \
  --output logs/benchmark.jsonl
```

The console prints per-trial summaries and a final aggregate (number of trials, parameter grids, output path).

## Testing

Unit tests cover preprocessing, geo filter bindings, positional/dynamics scoring, contact-surface logic, wrench math, trajectory builder, and the new accumulator:

```bash
python -m unittest discover -s tests
```

For native changes, add corresponding CTest targets inside `build` and run `ctest --test-dir build`.
