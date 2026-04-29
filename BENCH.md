# Benchmarking & Profiling

Headless benchmark frontend: `FractalGPUBench`.

It spins up a hidden GLFW window so the GPU path can register a real
`GL_RGBA32F` texture for CUDA-GL interop (falls back to host-upload if
registration fails — same code path as the main app). For each scenario it
runs `--warmup` warmup frames and `--frames` measured frames, advancing the
camera azimuth 1°/frame so rays don't repeat.

## Build

```sh
cmake --preset gpu-release
cmake --build build --target FractalGPUBench
```

Bench is built independently of the main app — the two share `renderer_cpu.cpp`
and `renderer_gpu.cu` but compile separately. Either presets `gpu-debug` or
`cpu-release` work too; CPU-only builds skip GPU scenarios at runtime.

## What it measures

Per scenario:

- `frame.mean / .median / .p99` — wall-clock around `renderer.render()`.
  End-to-end including the device→host `cudaMemcpy` on the fallback path.
- `kern.mean / .median / .p99` — GPU only; from `renderer.renderMs()`
  (cudaEvent, kernel + sync only — no readback).

Difference between `frame` and `kern` on the GPU = readback + interop overhead.

## Once-per-run GPU diagnostics

Printed at startup when GPU is enabled:

- Device name, CC, SM count, max threads/SM, warp size.
- Regs/SM, smem/SM, smem/block.
- Mem clock × bus → peak bandwidth.

For each kernel variant (surface + volumetric):

- `regs/thread`, static smem, dynamic smem, max threads/block.
- **Theoretical occupancy** at 16×16 block size = `active_blocks × 256 / max_threads_per_SM`.
- `cudaOccupancyMaxPotentialBlockSize` suggestion: the block size that would
  maximize occupancy for that kernel given its register/smem footprint.

## Reasonable runs

```sh
# Quick sanity (~30s, GPU + a tiny CPU baseline)
./build/FractalGPUBench --scenarios quick --tag $(git rev-parse --short HEAD)

# Full GPU sweep — 3 fractals × {low/med/hi/vol} = 12 scenarios
./build/FractalGPUBench --gpu --scenarios full --frames 100 \
    --tag $(git rev-parse --short HEAD) --csv bench.csv

# Sweep maxSteps (raymarcher cost scaling, fixed res)
./build/FractalGPUBench --gpu --scenarios steps --frames 100 \
    --tag $(git rev-parse --short HEAD)-steps --csv bench.csv

# Sweep volumetric step counts (smem footprint scaling)
./build/FractalGPUBench --gpu --scenarios vol --frames 100 \
    --tag $(git rev-parse --short HEAD)-vol --csv bench.csv

# Volumetric A/B: shared-mem two-pass vs. fused single-pass, multiple step counts
./build/FractalGPUBench --gpu --scenarios vol-compare --frames 100 \
    --tag $(git rev-parse --short HEAD)-volcmp --csv bench.csv

# CPU-only baseline (slow — single-threaded scalar code)
./build/FractalGPUBench --cpu --scenarios full --frames 20 \
    --tag $(git rev-parse --short HEAD) --csv bench.csv

# Force a custom resolution for every scenario
./build/FractalGPUBench --gpu --scenarios full --width 1920 --height 1080 \
    --frames 200 --tag fhd-$(git rev-parse --short HEAD) --csv bench.csv
```

### Try the interop fast path

On hybrid graphics (GL on iGPU, CUDA on dGPU) the bench falls back to
host-upload by default. Route GL onto the NVIDIA dGPU so the kernel can write
straight into the GL texture:

```sh
FRACTAL_FORCE_X11=1 __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia \
    ./build/FractalGPUBench --gpu --scenarios full --frames 100 \
    --tag interop-$(git rev-parse --short HEAD) --csv bench.csv
```

`frame.mean` should drop noticeably (no device→host memcpy); `kern.mean`
should stay roughly the same.

## Comparing two versions / branches

The CSV is append-mode and carries a `tag` column, so:

```sh
git checkout main
cmake --build build --target FractalGPUBench
./build/FractalGPUBench --gpu --scenarios full --frames 100 \
    --tag main-$(git rev-parse --short HEAD) --csv compare.csv

git checkout my-branch
cmake --build build --target FractalGPUBench
./build/FractalGPUBench --gpu --scenarios full --frames 100 \
    --tag mine-$(git rev-parse --short HEAD) --csv compare.csv
```

Sort by `(scenario, fractal, mode, width)` to line up matching rows across
tags.

## Deeper kernel profiling with Nsight

The bench prints theoretical occupancy. For **achieved** occupancy plus memory
throughput, branch divergence, instruction mix, etc., use Nsight Compute.

### Nsight Compute (per-kernel metrics)

Full report — slow (each kernel is replayed many times under instrumentation),
so use `--frames 1 --warmup 0`:

```sh
/usr/local/cuda-13/bin/ncu --set full --target-processes all \
    --kernel-name-base demangled \
    --kernel-regex 'kernel(Surface|Volumetric)' \
    -o ncu_report ./build/FractalGPUBench --scenarios quick --frames 1 --warmup 0
/usr/local/cuda-13/bin/ncu-ui ncu_report.ncu-rep
```

Lighter, terminal-friendly subset (achieved occupancy, throughput, branch
uniformity):

```sh
/usr/local/cuda-13/bin/ncu --set detailed \
    --kernel-regex 'kernel(Surface|Volumetric)' \
    --metrics sm__warps_active.avg.pct_of_peak_sustained_active,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
smsp__sass_average_branch_targets_threads_uniform.pct \
    ./build/FractalGPUBench --scenarios quick --frames 1 --warmup 0
```

Useful metrics to ask for individually:

| metric | what it means |
| --- | --- |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | achieved occupancy (vs. theoretical we already print) |
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | overall SM utilization |
| `dram__bytes.sum.per_second` | DRAM bytes/sec — compare against printed peak BW |
| `smsp__sass_average_branch_targets_threads_uniform.pct` | branch divergence (low = lots of divergence) |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` | global load sectors |
| `launch__registers_per_thread` | confirms what the bench printed |

### Nsight Systems (timeline)

Kernel launches, memcpys, CPU/GPU gaps:

```sh
/usr/local/cuda-13/bin/nsys profile -o bench_timeline \
    ./build/FractalGPUBench --gpu --scenarios quick --frames 50
/usr/local/cuda-13/bin/nsys-ui bench_timeline.nsys-rep
```

## Interpreting common findings

- **`frame.mean ≫ kern.mean`** → readback is the bottleneck. On hybrid laptops
  this is the host-upload fallback; switch to interop with the env vars above.
- **High theoretical occupancy but low achieved occupancy** (Nsight Compute) →
  warps are stalling on memory. Check `dram__bytes` and the L1/L2 hit rates.
  Ray-march workloads tend to be latency-hidden; if hit rates are low and
  throughput is bound, consider memory layout.
- **Low theoretical occupancy** → register or smem pressure. The bench prints
  `regs/thread` and `dyn_smem`; the volumetric kernel's smem-backed sampling
  cap is the typical culprit (see "Volumetric variants" below).
- **Branch uniformity below ~50%** → significant divergence. For raymarchers,
  this is usually rays that escape vs. hit at very different step counts; not
  much you can do without splitting work.

## Volumetric variants (shared-memory vs. fused)

Two implementations of the volumetric raymarcher are available, picked at
runtime via `RenderParams::vol.useSharedMem`:

- **smem (default)** — `kernelVolumetric`. Two-pass: sample density+trap into
  per-thread dynamic SMEM (`block.x * block.y * vol_steps * 8` bytes), then
  composite front-to-back. Smem footprint caps active blocks per SM, lowering
  theoretical occupancy.
- **fused** — `kernelVolumetricFused`. Single-pass sample + composite, with
  early-out when `accAlpha >= 0.99`. No dynamic SMEM, so occupancy isn't
  smem-bound. Skips remaining SDF evaluations after alpha saturates — should
  win on dense scenes; near-tie on thin/empty ones.

Toggle from the UI ("Use shared mem" checkbox under Render Mode → Volumetric)
or run the `vol-compare` scenario set in the bench. The bench's startup
diagnostics report both kernels' regs/thread, smem, and theoretical occupancy.
