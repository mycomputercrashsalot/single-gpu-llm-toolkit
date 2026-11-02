# Legacy GPU Support (MI50 / gfx906 and Older)

## Overview

The MI50 (gfx906) and MI100 (gfx908) are deprecated in ROCm 7.x. The Linux
kernel driver (`amdgpu.ko`) still supports them, but the user-space ROCm
libraries (rocBLAS, hipCUB, rocPRIM, etc.) **no longer ship pre-built kernels
for gfx906/gfx908 targets**. This means code that compiles fine on an MI300
may fail at runtime on an MI50 with errors like:

```
rocBLAS ERROR: TensileLibrary file not found for gfx906
HIP error: no kernel image is available for execution on the device
```

This document explains what works, what doesn't, and how to get this codebase
running on MI50/MI100 hardware.

---

## What Still Works

| Component | MI50 (gfx906) | MI100 (gfx908) |
|-----------|---------------|----------------|
| Kernel driver (`amdgpu.ko`) | Yes | Yes |
| `rocminfo` / `rocm-smi` | Yes | Yes |
| HIP runtime (`hipRuntime`) | Partial | Yes |
| `hipcc` compiler | Yes | Yes |
| rocBLAS (pre-built) | **No** (dropped in ROCm 7) | **No** (dropped in ROCm 7) |
| hipCUB (pre-built) | **No** | **No** |
| MIOpen | **No** | Partial |
| `hipDeviceCanAccessPeer` (P2P) | **Broken** (see #4793) | Yes |

**Key issue:** The pre-built ROCm packages from `repo.radeon.com` no longer
include gfx906/gfx908 library targets. The kernel driver and HIP runtime still
work, but the compute libraries need to be built from source.

---

## ROCm Version Guidance

| ROCm Version | gfx906 (MI50) | gfx908 (MI100) | Notes |
|-------------|---------------|----------------|-------|
| 5.x | Full support | Full support | Last version with complete legacy support |
| 6.x | Partial support | Partial support | Some libraries dropped gfx906 |
| 7.0 | Deprecated | Deprecated | No pre-built gfx906/908 targets |
| 7.1+ | Deprecated | Deprecated | Only MI250+ officially supported |

**Recommendation:** For MI50/MI100, use ROCm 6.x if possible. If you must use
ROCm 7.x+, you need to build the libraries from source (see below).

---

## Building rocBLAS from Source with gfx906 Support

The most common failure point is rocBLAS not having gfx906 TensileLibrary
files. To fix this, build rocBLAS from source targeting gfx906:

```bash
# Clone rocBLAS
git clone https://github.com/ROCm/rocBLAS.git
cd rocBLAS

# Build for gfx906 (MI50) and gfx908 (MI100)
cmake -B build \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DAMDGPU_TARGETS="gfx906;gfx908" \
  -DBUILD_CLIENTS_CMAKE=OFF \
  -DBUILD_DEV=OFF

cmake --build build -j$(nproc)
sudo cmake --install build
```

Similarly for hipCUB:

```bash
git clone https://github.com/ROCm/hipCUB.git
cd hipCUB
cmake -B build \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DCMAKE_HIP_ARCHITECTURES="gfx906;gfx908"
cmake --build build -j$(nproc)
sudo cmake --install build
```

---

## Runtime Workaround: HSA_OVERRIDE_GFX_VERSION

If you can't rebuild rocBLAS, you can try the `HSA_OVERRIDE_GFX_VERSION`
environment variable to force the ROCm runtime to use kernels compiled for a
different (but compatible) architecture. This is a best-effort workaround:

```bash
# On MI50 (gfx906), try overriding to gfx90a (MI250) — same GCN 5th gen family
export HSA_OVERRIDE_GFX_VERSION=90a

# Or try gfx900 (Vega) if that's available in your rocBLAS install
export HSA_OVERRIDE_GFX_VERSION=900
```

**Warning:** This is not guaranteed to work. Wrong overrides can cause silent
numerical errors or crashes. The `rocm_env.py` script in this repo handles
this safely by only overriding within the same `gfxNN` major family.

```python
# In your Python training script, call BEFORE importing torch:
from rocm_env import setup_rocm_env
info = setup_rocm_env()
print(info)  # see what override was applied (if any)
import torch
```

---

## Known Limitations on MI50

1. **Peer-to-peer (P2P) is broken** — See [ROCm issue #4793](https://github.com/ROCm/ROCm/issues/4793).
   Multi-GPU P2P between MI50 cards does not work in ROCm 7.x.

2. **rocBLAS missing kernels** — Many rocBLAS routines will fall back to CPU
   or fail at runtime if gfx906 TensileLibrary files are absent.

3. **FP8 not supported** — MI50 does not have native FP8 compute. The codebase
   correctly disables FP8 for gfx906:
   ```python
   # work/single-gpu-llm-toolkit/backends/rocm.py
   return arch.startswith(("gfx942", "gfx950", "gfx95", "gfx12"))
   ```

4. **No XGMI** — MI50 uses PCIe only (no Infinity Fabric/XGMI). Multi-GPU
   scaling will be limited by PCIe bandwidth.

---

## Step-by-Step: Getting Sirius Running on MI50

1. **Verify the kernel driver sees the GPU:**
   ```bash
   rocm-smi --showinfo
   rocminfo | grep -i gfx
   # Should show gfx906
   ```

2. **Check your ROCm version:**
   ```bash
   rocm-smi --showrocmversion
   # ROCm 6.x is preferred; 7.x+ requires source builds
   ```

3. **Build rocBLAS from source** (if on ROCm 7.x+):
   See "Building rocBLAS from Source" section above.

4. **Build Sirius for gfx906:**
   ```bash
   cd sirius-rocm
   ./scripts/setup.sh
   ./scripts/build_rocm_deps.sh
   # Force gfx906 build target
   cmake -B build -S . -DCMAKE_HIP_ARCHITECTURES=gfx906
   cmake --build build -j$(nproc) --target duckdb sirius_extension
   ```

5. **Run with override if needed:**
   ```bash
   HSA_OVERRIDE_GFX_VERSION=90a ./run_tpch_rocm.sh
   ```

---

## Step-by-Step: Getting LLM Training Working on MI50

1. **Set up ROCm environment:**
   ```bash
   cd work/single-gpu-llm-toolkit
   python3 rocm_env.py --selftest
   ```

2. **Check if your PyTorch ROCm wheel supports gfx906:**
   ```bash
   python3 -c "import torch; print(torch.cuda.get_arch_list())"
   # Look for 'gfx906' in the output
   ```

3. **If gfx906 is missing**, use the override:
   ```bash
   # This will auto-detect and set HSA_OVERRIDE_GFX_VERSION if needed
   python3 train_sft.py --gfx-override gfx90a ...
   ```

4. **If kernels still fail**, you need a PyTorch wheel compiled with gfx906:
   ```bash
   # Build PyTorch from source with gfx906 support
   export TORCH_CUDA_ARCH_LIST="9.0"
   export HSA_OVERRIDE_GFX_VERSION=906
   pip install --no-build-isolation -e .
   ```

---

## Related ROCm Issues

- [#5215](https://github.com/ROCm/ROCm/issues/5215) — Feature request to restore MI50/MI60 support
- [#4793](https://github.com/ROCm/ROCm/issues/4793) — MI50 32GB P2P not working
- [#4625](https://github.com/ROCm/ROCm/issues/4625) — rocBLAS missing gfx906 TensileLibrary
- [#6176](https://github.com/ROCm/ROCm/issues/6176) — Kernel driver issues with MI50
- [#2849](https://github.com/ROCm/ROCm/issues/2849) — Documentation: deprecated gfx906 label

---

## Summary

| Question | Answer |
|----------|--------|
| Is MI50 supported by the kernel driver? | Yes (`amdgpu.ko`) |
| Is MI50 supported by ROCm 7.x libraries? | No (deprecated, no pre-built targets) |
| Can we still use MI50? | Yes, with source builds + runtime overrides |
| Does P2P work on MI50? | No (known issue #4793) |
| Does this codebase support MI50? | Build targets added; runtime depends on ROCm libraries |
