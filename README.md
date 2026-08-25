# xfeat-cuda-graph

CUDA Graph acceleration for [XFeat (CVPR 2024)](https://github.com/verlab/accelerated_features) and LightGlue matcher.

Capture the native PyTorch forward pass **once** in GPU memory, then replay with zero kernel-launch overhead. Output is **bit-identical** to the eager torch path.

## Why?

On Windows WDDM (and to a lesser extent Linux), PyTorch launches hundreds of tiny kernels for XFeat CNN and LightGlue transformers. Each kernel launch carries ~5-20 µs of driver overhead. CUDA Graph captures the entire GPU kernel graph once and replays it as a single unit — removing all launch overhead.

### Measured speedups (RTX 4090 Laptop, Windows WDDM)

| Component | Resolution / Budget | Eager | CUDA Graph | Speedup |
|---|---|---|---|---|
| XFeat CNN | 640×640 | 7.0 ms | 1.4 ms | **5.0×** |
| LightGlue B=1 | M=512, N=512 | 16.0 ms | 1.75 ms | **9.1×** |

## Setup (conda)

```bash
# 1. Create a fresh conda environment
conda create -n xfeat-cg python=3.10 -y
conda activate xfeat-cg

# 2. Install PyTorch (CUDA 12.4)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 3. Install remaining dependencies
pip install kornia jupyter matplotlib opencv-python

# 4. Install the submodule accelerated_features/
git submodule update --init --recursive
```

## Quick start

```python
from helper import (
    patch_kornia_capture_safe,
    CGLightGlue, capture_cg_lightglue,
    CGXFeatNet, capture_cg_xfeat, xfeat_cg_extract,
)

# 1. LightGlue CUDA Graph
patch_kornia_capture_safe()
model = CGLightGlue().cuda().eval()
cg = capture_cg_lightglue(model, M=512, N=512)
cg["graph"].replay()           # ~1.75ms vs ~16ms eager

# 2. XFeat CNN CUDA Graph
from accelerated_features.modules.xfeat import XFeatModel
xfeat_net = XFeatModel().cuda().eval()
cg_net = CGXFeatNet(xfeat_net, H=704, W=640).cuda().eval()
xcg = capture_cg_xfeat(cg_net, H=704, W=640)
kpts, desc, scores = xfeat_cg_extract(xcg, rgb_tensor, top_k=512)
```

## Project structure

```
xfeat-cuda-graph/
├── helper.py                              # Core CUDA Graph wrappers
├── accelerated_features/                  # git submodule (XFeat model)
│   ├── modules/
│   │   ├── xfeat.py
│   │   ├── lighterglue.py
│   │   └── model.py
│   └── weights/
│       ├── xfeat.pt
│       └── xfeat-lighterglue.pt
├── 01_xfeat_cudagraph_benchmark       # XFeat: resolution sweep + bit-exactness
├── 02_lightglue_cudagraph_benchmark   # LG: sweep B, M, N → speedup
├── 03_end_to_end_pipeline             # Full pipeline: XFeat + LG, eager vs CG
└── README.md
```

## Notebooks

### 01 — XFeat CUDA Graph Benchmark
Tests XFeat CNN at multiple resolutions. Verifies bit-exactness between eager and CG output. Visualises keypoint overlay.

### 02 — LightGlue CUDA Graph Benchmark
Sweeps batch sizes (B∈{1,2,4,8,16}) and point budgets (M,N∈{64,128,256,512}).
Compares eager vs CUDA Graph latency, reports speedup table.

### 03 — End-to-End Pipeline
Full pipeline: XFeat extraction → LightGlue matching. Compares total latency and match quality between eager and CG-accelerated paths.

## License

MIT — same as the original [accelerated_features](https://github.com/verlab/accelerated_features) project.