<div align="center">
<h1>RobustHOI: Robust Hand-object Reconstruction from
RGB-D Video by Leveraging Hand and Object Priors</h1>
</div>

## Overview

Reconstructing hand-object interactions (HOI) from RGB-D video is challenging: objects are often textureless, reflective, or tiny, and mutual hand-object occlusion leaves large surface regions unobserved. RobustHOI addresses this by jointly leveraging generative shape priors, multi-view geometric constraints, and hand-object contact priors, producing complete and metrically accurate reconstructions of both object and hand across all frames. RobustHOI achieves a **success rate exceeding 95%** on a challenging dataset of 108 in-the-wild HOI sequences, demonstrating robustness well beyond controlled benchmarks.

## System Requirements

This setup was verified on:

- **OS**: Ubuntu 22.04, x86_64
- **GPU**: NVIDIA RTX 4090 (Ada, compute capability `sm_89`)
- **CUDA toolkit**: 12.4 (`nvcc` at `/usr/local/cuda-12.4`); NVIDIA driver supporting CUDA ≥ 12.1
- **Host compiler**: gcc/g++ 11 (required by `nvdiffrast`/`tiny-cuda-nn`)
- **Python**: 3.10
- **Package manager**: [`uv`](https://github.com/astral-sh/uv) (no conda required)

> **Note on PyTorch CUDA build.** PyTorch is installed as `2.1.0+cu121`. The `cu121`
> runtime wheels run fine against the CUDA 12.4 driver/toolkit, and the locally
> compiled extensions (`nvdiffrast`, `tiny-cuda-nn`, FoundationPose `mycuda`) are
> built with the system's CUDA 12.4 `nvcc`.

> **Adapt for your GPU.** Replace `sm_89` / `8.9` / `TCNN_CUDA_ARCHITECTURES=89`
> below with your GPU's compute capability (e.g. `86` for RTX 3090, `80` for A100).

## Installation

### 1. Clone with submodules

```bash
git clone --recurse-submodules git@github.com:byran-wang/RobustHOI.git
cd RobustHOI
git submodule update --init --recursive
```

### 2. Create the environment and install PyTorch

```bash
uv venv --python 3.10 .venv
source .venv/bin/activate

# PyTorch 2.1.0 + CUDA 12.1. The official index (download.pytorch.org/whl/cu121)
# works but can be slow; the Aliyun mirror below is a fast drop-in in CN.
uv pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    --find-links https://mirrors.aliyun.com/pytorch-wheels/cu121/ \
    --index-url https://mirrors.aliyun.com/pypi/simple/

# Two important pins:
#   numpy<2       -> torch 2.1.0 is built against the NumPy 1.x ABI
#   setuptools<81 -> torch's cpp_extension imports pkg_resources, removed in setuptools 81+
uv pip install "numpy==1.23.1" "setuptools==69.5.1" wheel

# Sanity check
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# -> 2.1.0+cu121 True NVIDIA GeForce RTX 4090
```

> If `torch.cuda.is_available()` is `True` but you later hit `libcublas`/`libcudnn`
> import errors, install the CUDA runtime wheels explicitly:
> ```bash
> uv pip install nvidia-cublas-cu12==12.1.3.1 nvidia-cudnn-cu12==8.9.2.26 \
>   nvidia-cuda-runtime-cu12==12.1.105 nvidia-cuda-nvrtc-cu12==12.1.105 \
>   nvidia-cuda-cupti-cu12==12.1.105 nvidia-cufft-cu12==11.0.2.54 \
>   nvidia-curand-cu12==10.3.2.106 nvidia-cusolver-cu12==11.4.5.107 \
>   nvidia-cusparse-cu12==12.1.0.106 nvidia-nccl-cu12==2.18.1 nvidia-nvtx-cu12==12.1.105 \
>   --find-links https://mirrors.aliyun.com/pytorch-wheels/cu121/
> ```

### 3. Install Python requirements

```bash
# chumpy has a broken build (imports numpy/pip at build time) -> install it separately
uv pip install -r <(grep -v '^chumpy' requirements.txt) \
    --index-url https://mirrors.aliyun.com/pypi/simple/
uv pip install chumpy==0.70 --no-build-isolation \
    --index-url https://mirrors.aliyun.com/pypi/simple/
```

### 4. Build the compiled extensions

```bash
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export CC=/usr/bin/gcc CXX=/usr/bin/g++
mkdir -p dependency

# --- utils_simba (logging / rendering / depth helpers, imported everywhere) ---
# Its setup.py packaging is broken; expose it on the path directly:
echo "$(pwd)/third_party/utils_simba" > .venv/lib/python3.10/site-packages/utils_simba.pth
python -c "from utils_simba.logger import get_logger; print('utils_simba OK')"

# --- nvdiffrast (FoundationPose rasterization; JIT-compiles CUDA on first use) ---
git clone --depth 1 https://github.com/NVlabs/nvdiffrast.git dependency/nvdiffrast
uv pip install -e dependency/nvdiffrast --no-deps --no-build-isolation

# --- PyTorch3D (prebuilt wheel for py310 / cu121 / pyt210) ---
uv pip install \
  https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt210/pytorch3d-0.7.5-cp310-cp310-linux_x86_64.whl \
  --no-deps
uv pip install iopath   # runtime dependency of pytorch3d.io

# --- smplx (custom fork used by the hand model) ---
git clone --depth 1 https://github.com/zc-alexfan/smplx.git dependency/smplx
uv pip install -e dependency/smplx --no-deps --no-build-isolation

# --- tiny-cuda-nn (NeuS hash encoding) ---
git clone --recursive https://github.com/NVlabs/tiny-cuda-nn.git dependency/tiny-cuda-nn
TCNN_CUDA_ARCHITECTURES=89 python dependency/tiny-cuda-nn/bindings/torch/setup.py install

# --- Eigen 3.4.0 headers (header-only; needed by FoundationPose mycuda/mycpp) ---
wget -q https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz -O /tmp/eigen-3.4.0.tar.gz
tar -xzf /tmp/eigen-3.4.0.tar.gz -C dependency/
export EIGEN_DIR="$(pwd)/dependency/eigen-3.4.0"

# --- FoundationPose CUDA extension (octree ray tracing) ---
( cd third_party/FoundationPose/bundlesdf/mycuda && \
  TORCH_CUDA_ARCH_LIST="8.9" CPATH="$EIGEN_DIR:$CPATH" \
  python -m pip install -e . --no-build-isolation )

# --- FoundationPose C++ extension mycpp (pose clustering) ---
# The repo's CMakeLists requires Boost, but the sources only #include'd Boost
# headers that are never actually used. This build path drops Boost and points
# CMake at the local Eigen headers (no system Boost / no sudo required).
uv pip install cmake
( cd third_party/FoundationPose/mycpp && rm -rf build && mkdir build && cd build && \
  cmake .. -Dpybind11_DIR="$(python -c 'import pybind11; print(pybind11.get_cmake_dir())')" \
           -DEIGEN3_INCLUDE_DIR="$EIGEN_DIR" && \
  make -j$(nproc) )
```

If `find_package(Boost REQUIRED ...)` is still present in
`third_party/FoundationPose/mycpp/CMakeLists.txt`, remove the Boost lines and the
three unused `#include <boost/...>` lines in `mycpp/include/Utils.h` and
`mycpp/src/app/pybind_api.cpp` (the patch is already applied in this checkout).

### 5. Interpreter shim for `run_rhoi.py`

`run_rhoi.py` launches each pipeline stage as a subprocess using the hard-coded path
`~/miniconda3/envs/robust_hoi/bin/python` (and the conda gcc names for `CC`/`CXX`).
Point those at the uv venv so no code changes are needed:

```bash
mkdir -p ~/miniconda3/envs
ln -sfn "$(pwd)/.venv" ~/miniconda3/envs/robust_hoi
ln -sf "$(which gcc)" .venv/bin/x86_64-conda-linux-gnu-gcc
ln -sf "$(which g++)" .venv/bin/x86_64-conda-linux-gnu-g++
```

(Symlinking the **whole** venv directory — not just `bin/python` — is required so
that Python finds `pyvenv.cfg` and resolves the venv's `site-packages`.)

### 6. Model weights & assets

These are not in the repo and must be placed manually:

| Asset | Location | Notes |
|-------|----------|-------|
| FoundationPose weights | `third_party/FoundationPose/weights/{2023-10-28-18-33-37, 2024-01-11-20-02-45}/` | scorer + refiner checkpoints (`model_best.pth`, `config.yml`), ~247 MB |
| MANO / contact models | `body_models/` (repo root) | `MANO_RIGHT.pkl`, `contact_zones.pkl`, `sealed_vertices_sem_idx.npy`, … (~257 MB) |
| HaMeR MANO data | `third_party/hamer/_DATA/data/mano/` | `cp -r body_models/* third_party/hamer/_DATA/data/mano` (only needed for hand-pose estimation stages) |

## Data

This repo was verified on **HO3D_v3**. Point the repo at your dataset directory with a
symlink at the repo root (several eval scripts read `./ho3d_v3/...` relative to the repo):

```bash
ln -sfn /path/to/HO3D_v3 ho3d_v3
# e.g. ln -sfn ~/Documents/dataset/BundleSDF/HO3D_v3 ho3d_v3
```

`confs/sequence_config_ho3d.py` resolves the dataset at
`~/Documents/dataset/BundleSDF/HO3D_v3/train/` when `DATASET=ho3d`. Expected layout:

```
HO3D_v3/
├── train/{seq}/             # rgb/, depth/, meta/, mask_*/, plus precomputed
│                            #   pipeline_preprocess/, pipeline_corres/,
│                            #   SAM3D_aligned_post_process/, SAM3D_align_filter/
├── models/{object_id}/      # YCB 3D models (textured.obj)
└── processed/{seq}.pt       # preprocessed GT (used by vggt/utils/gt.py)
```

The end-to-end script below consumes the already-preprocessed intermediates
(`pipeline_preprocess/`, `pipeline_corres/`, `SAM3D_aligned_post_process/`). To generate
those from raw RGB-D (SAM3 masks, SAM3D shape prior, HaMeR hand pose, FoundationStereo
depth, etc.), see the full stage list in `CLAUDE.md` / `run_rhoi.sh`.

## Running

The verified entry point reconstructs object + hand for a sequence end-to-end:

```bash
bash run_rhoi_ho3d.sh          # default seq_list="MC1"
```

The script activates the venv, sets `DATASET=ho3d`, and runs, in order:

1. `hoi_pipeline_joint_opt`   — PnP/RANSAC registration + FoundationPose tracking + keyframe bundle adjustment + incremental NeuS
2. `hoi_pipeline_neus_global` — global NeuS surface reconstruction
3. `hoi_pipeline_align_hand_object_{h,r,o,ho}` — hand / rotation / object / hand+object alignment
4. `hoi_pipeline_eval`        — object pose (ADD/ADD-S), shape (CD/F-score), hand (MPJPE) metrics
5. `eval_sum`                 — aggregate the per-sequence table

Outputs land under `output/{seq}/`:

```
output/MC1/
├── pipeline_joint_opt/        # per-frame results, neus_data/, txt.log
├── pipeline_neus_global/      # global NeuS mesh
├── align_hand_object/         # aligned hand + object
└── pipeline_joint_opt/eval/   # metric.json, metric_all.npy
output/metrics_summary/eval.txt # aggregated table
```

To run other sequences, edit `seq_list` in `run_rhoi_ho3d.sh` (19 HO3D sequences are
configured in `confs/sequence_config_ho3d.py`).

## Repository Layout

- `run_rhoi.py` — central orchestrator mapping `--execute_list` / `--process_list` to stages
- `run_rhoi_ho3d.sh` — minimal end-to-end HO3D run; `run_rhoi.sh` — the full pipeline incl. preprocessing
- `robust_hoi_pipeline/` — the core pipeline (joint optimization, NeuS integration, evaluation)
- `confs/` — dataset/sequence configuration (`DATASET` env var selects the dataset)
- `third_party/` — FoundationPose, instant-nsr-pl (NeuS), SAM3, HaMeR, utils_simba, …
- See `CLAUDE.md` for an in-depth architecture and coordinate-convention reference.

## License

See the [LICENSE](./LICENSE.txt) file for details about the license under which this code is made available.
