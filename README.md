# MD_CUDA_refactored

CUDA/C++ molecular dynamics code with Lennard-Jones and neural-network
potential backends.

## Dependencies

- CUDA toolkit with `nvcc`
- CMake 3.18 or newer
- A C++17 compiler supported by CUDA
- Python development headers
- PyTorch/libtorch with CMake package files
- Optional: ONNX Runtime, only if an ONNX backend is added/enabled
- [nlohmann/json](https://github.com/nlohmann/json), vendored under `include/external`

## Build

Configure by pointing CMake to your local PyTorch CMake package. With a Python
wheel install of PyTorch, `Torch_DIR` is usually under the Python environment's
`site-packages/torch/share/cmake/Torch`.

```sh
cmake -G Ninja -S . -B build \
  -D CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -D Torch_DIR=/path/to/site-packages/torch/share/cmake/Torch \
  -D MD_CUDA_ARCHITECTURES=86
ninja -C build
```

Alternatively, set `LIBTORCH_PATH` to the PyTorch package/root and CMake will
look for `share/cmake/Torch` below it.

```sh
cmake -G Ninja -S . -B build \
  -D CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -D LIBTORCH_PATH=/path/to/site-packages/torch \
  -D MD_CUDA_ARCHITECTURES=86
```

If ONNX Runtime is needed by future code paths, add:

```sh
-D ONNXRUNTIME_ROOT=/path/to/onnxruntime
```

## Run

The executable takes one JSON workflow file:

```sh
./build/MD_MLP configs/example_workflow_bussi.json
```

For a quick GPU smoke test, run:

```sh
./build/MD_MLP configs/smoke_lj_nve.json
```

## Hydra test environment

On `hydra`, the repository can be tested under:

```text
/home/emi/workspace/MD_CUDA_refactored_codex
```

The installed NVIDIA driver reports CUDA 12.2 support, so use the CUDA 12.2
toolkit instead of the `/usr/local/cuda` symlink if that points at a newer
toolkit:

```sh
CUDA_HOME=/usr/local/cuda-12.2 CUDA_PATH=/usr/local/cuda-12.2 \
cmake -G "Unix Makefiles" -S . -B build-codex \
  -D CMAKE_C_COMPILER=/opt/rh/devtoolset-11/root/usr/bin/gcc \
  -D CMAKE_CXX_COMPILER=/opt/rh/devtoolset-11/root/usr/bin/g++ \
  -D CMAKE_CUDA_COMPILER=/usr/local/cuda-12.2/bin/nvcc \
  -D CMAKE_CUDA_HOST_COMPILER=/opt/rh/devtoolset-11/root/usr/bin/g++ \
  -D CUDAToolkit_ROOT=/usr/local/cuda-12.2 \
  -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.2 \
  -D Torch_DIR=/home/emi/miniforge3/envs/edamame_md/lib/python3.13/site-packages/torch/share/cmake/Torch \
  -D MD_CUDA_ARCHITECTURES=75
cmake --build build-codex -j2
./build-codex/MD_MLP configs/smoke_lj_nve.json
```

## JSON workflow overview

Top-level keys:

- `meta`: `name`, `unit` (`lj` or `metal`), and random `seed`
- `common_settings`: atoms, cell, neighbour list, and potential settings
- `steps`: one or more simulation or minimization steps

Supported atom initialization:

- `generate_binary_lj`
- `from_file` with `format: "xyz"`

The XYZ reader expects an extended XYZ `Lattice="..."` comment and atom lines in
one of these forms:

```text
species x y z fx fy fz
species x y z
```

When force columns are omitted, initial forces are set to zero. Only the first
frame is read from a multi-frame XYZ file.

Supported cell:

- `cubic`

Supported potentials:

- `lennard_jones`
- `NNP`
- `NNP_csr`
- `NNP_fixed`
- `NNP_aoti`

Supported ensembles:

- `NVE`
- `NVT` with `Nose-Hoover`, `Bussi`, or `Langevin`

Supported minimizer:

- `fire`

Each step should contain `observer`. The legacy key `output` is still accepted
as an alias. Set `"step": "reset"` inside a step to reset the step counter
before that step starts. Velocities are initialized only once by default; set
`"initialize_velocities": true` in a step's ensemble to force reinitialization,
or `false` to keep existing velocities.

Thermal degrees of freedom and COM drift handling can be set per ensemble step.
By default, both temperature reporting and thermostats use the legacy `3N`
definition and no in-run COM drift removal is applied. For PBC bulk runs where
the whole-system COM motion is removed, use:

```json
"ensemble": {
  "type": "NVT",
  "temperature": 2500.0,
  "thermostat": "Nose-Hoover",
  "tau": 1.0,
  "temperature_dof": "3N-3",
  "thermostat_dof": "3N-3",
  "rescale_initial_temperature": true,
  "remove_com_drift_interval": 128
}
```

`temperature_dof` controls reported temperature, initial-temperature rescaling,
and target-temperature export checks. `thermostat_dof` controls the Nose-Hoover
and Bussi thermostat target kinetic energy. `remove_com_drift_interval` removes
the mass-weighted whole-system COM velocity after completed MD steps; it requires
`use_graph=false`.

`use_graph=true` enables CUDA Graph capture for graph-safe interactions. It is
currently supported for Lennard-Jones interactions only. Use `use_graph=false`
for `NNP`, `NNP_csr`, `NNP_fixed`, and `NNP_aoti`.

Supported observers include:

- `linear`: print energies every fixed number of steps
- `log`: print energies on a simple logarithmic time grid
- `linear_export_trajectory`: write extxyz trajectory frames every fixed number
  of steps
- `log_export_trajectory`: write one extxyz trajectory frame at each simple
  logarithmic time point
- `dense_log_burst_export_trajectory`: write an MSD/time-average friendly
  trajectory with dense early sampling followed by log-spaced anchors and short
  fixed-interval bursts
- `target_temperature_export`: write structures when a linear temperature
  schedule crosses target temperatures

Trajectory-writing observers accept an optional field-selection block:

```json
"trajectory": {
  "mode": "vdos",
  "fields": ["position", "velocity"],
  "coordinates": "wrapped",
  "format": "extxyz"
}
```

`fields`, when present, overrides the preset field list.  Species is always
written.  Supported fields are `position`, `velocity`, `force`, and the global
`energy`.  The presets are:

- `legacy`: position, force, and energy; preserves `is_unwrap`
- `msd`: unwrapped position
- `vdos`: wrapped position and velocity
- `active_learning`: wrapped position, force, and energy
- `transport_base`: wrapped position, velocity, force, and energy

`vdos` and `transport_base` require `linear_export_trajectory`, because VACF,
vDOS, and transport postprocessing require uniform time spacing.  Explicit
trajectory blocks add `time_fs` and field-unit metadata.  Without a trajectory
block, the previous position/force/energy extxyz schema and `is_unwrap`
behavior are retained.  `transport_base` is a trajectory input for later
postprocessing; velocity output alone does not provide the model-internal
semi-local heat-flux terms required for a complete Green-Kubo implementation.

After running `configs/smoke_ns2_painn_output_modes.json`, validate the field
schemas, uniform time grid, reconstructed initial temperature, and finite
VACF/FFT diagnostics with:

```sh
python scripts/validate_trajectory_output_modes.py \
  --vdos outputs/trajectory/smoke_ns2_vdos.xyz \
  --msd outputs/trajectory/smoke_ns2_msd.xyz \
  --transport-base outputs/trajectory/smoke_ns2_transport_base.xyz \
  --expected-initial-temperature-k 300
```

For `dense_log_burst_export_trajectory`, a T3400K-style setup is:

```json
"observer": {
  "type": "dense_log_burst_export_trajectory",
  "output_path": "./outputs/trajectory/MSD_T3400_style.xyz",
  "is_unwrap": true,
  "N_per_decade": 5,
  "M_burst": 10,
  "interval_burst": 10,
  "dense_until": "auto",
  "write_metadata": true
}
```

With `dense_until: "auto"`, the code emits every early step until adjacent
log-spaced burst windows no longer overlap. Later frames contain extxyz comment
metadata such as `step_rel`, `step_abs`, `time_fs`, `sample_type`, `burst_id`,
and `burst_idx`, which makes time-window grouping easier in downstream MSD or
VACF analysis. Keep `use_graph=false` for dense or burst trajectory sampling so
the observer is called every MD step.

## Neural-network potential interface

`NNP` and `NNP_aoti` expect models with inputs:

- `x`: `int64[N]` atomic numbers
- `edge_index`: `int64[2, E]`
- `edge_weight`: `float32[3, E]` relative displacement vectors

`NNP_csr` is the variable-shape MD CSR backend and additionally passes:

- `offsets`: `int64[N + 1]`

Export a compatible four-input PaiNN artifact with
`--forward-mode md --aggregation-mode csr`. Valid edges are grouped by source
atom, and the exported wrapper uses CSR segment reductions for scalar and
vector message aggregation. The runtime narrows edge tensors to the current
edge count before each call, so unused capacity is not evaluated by PaiNN.

`NNP_fixed` is retained as an experimental compatibility backend. It uses the
ordinary three-input NNP model with fixed-capacity edge
tensors.  Valid edges are followed by cutoff-zeroed padding edges, so the
steady-state force loop avoids the per-step device-to-host edge-count copy.
Padding self edges are distributed across atoms rather than all targeting atom
zero, avoiding a `scatter_add` atomic hotspot while retaining zero contribution.
The first graph reports its edge count, and later capacity overflow fails on
the CUDA stream before truncated forces can be integrated.  Calibrate with the
ordinary `NNP` backend first and set `max_edges` above the reported maximum;
20 percent headroom is recommended for NVT production.

The PaiNN exporter also contains opt-in research paths for shared geometry and
MD_CUDA paired-edge folding (`--geometry-mode shared` and
`--edge-layout md_cuda_paired`).  They preserve the existing three-input
TorchScript ABI, but are not production backends: the Gen6 production400
checkpoint exceeds the strict `1e-5 eV/Angstrom` maximum-force parity gate for
both paths.  `scripts/validate_painn_paired_parity.py` records the gate result,
and `scripts/benchmark_painn_compile_feasibility.py` stops before compilation
when the functional AOTAutograd force path fails the same gate.  The default
directed, per-layer MD export remains unchanged.

All NNP variants expect a tuple-like output:

```text
(energy, forces)
```

where `energy` is a scalar tensor and `forces` is laid out as x/y/z components
for all atoms. The runtime validates `max_edges` and output tensor shapes before
copying forces.

The AOTI example uses relative placeholders:

- `data/sample_NS2.xyz`
- `models/model_schnet_aoti.pt2`

Place the corresponding structure and model files there before running
`configs/example_workflow_NVE_NNP_aoti.json`.

### Exporting simplegnn PaiNN for `NNP`

The PaiNN model in `eminamitani/simplegnn_version2` has a Python training
interface with `batch` and returns forces as `[N, 3]`. The MD `NNP` backend
expects a TorchScript model with three inputs and forces laid out as x/y/z
blocks. Export a compatible wrapper with:

```sh
python scripts/export_simplegnn_painn_for_md.py \
  --simplegnn-root /path/to/simplegnn_version2 \
  --checkpoint /path/to/painn_model.pth \
  --output models/deployed_painn_model.pt \
  --natom-basis 60 \
  --n-radial 40 \
  --cutoff 5.0 \
  --epsilon 1e-7 \
  --num-interactions 2 \
  --radial-type gauss \
  --envelope-type smoothstep
```

New exports use the MD-only PaiNN forward path by default.  It preserves the
three-input `(x, edge_index, edge_weight)` ABI and returns the same total energy
and forces while skipping batch aggregation, virial construction, and the
retention of the higher-order training graph after the force derivative.  The
derivative itself uses PaiNN's legacy `create_graph=True` path because some
models exceed the strict force-parity tolerance with the alternate backward
kernel selected by `create_graph=False`; it is detached immediately.  Pass
`--forward-mode legacy` to export the former training-forward wrapper for parity
or regression checks.

Then set the potential type to `NNP` and point `model_path` to the exported
`.pt` file.

For the variable-shape CSR path, add `--aggregation-mode csr` and set the
potential type to `NNP_csr`. The default remains `--aggregation-mode scatter`
with the ordinary three-input `NNP` backend. CSR aggregation is available only
with `--forward-mode md`.

## Cell list

`cell_list` can be either a boolean or an object:

```json
"cell_list": {
  "enabled": true,
  "divisions": 6
}
```

The cell-list path is currently enabled only for Lennard-Jones with a cubic
cell. If `divisions` is omitted, it is inferred from `Lbox / (cutoff + margin)`.
The cell width must be at least `cutoff + margin`, and at least three divisions
are required to avoid duplicated periodic neighbor cells.
