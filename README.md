# Forge

Forge is a CPU-native kernel compiler in Rust that takes a small kernel language through typed SSA, fixed-point optimization, and AVX code generation, then emits inspectable Rust source instead of opaque binaries. The project targets the practical gap between hand-tuned CPU kernels and GPU-first compiler stacks by keeping the whole compilation and execution model on host CPUs, with explicit IR verification, deterministic runtime semantics, and no GPU driver dependency.

## Performance

Benchmarks run on commit `9d8c4b0`, 11th Gen Intel Core i3-1115G4 (Tiger Lake),
4 threads, turbo disabled (`no_turbo=1`), WSL2. Metric is median of 10 runs,
each run using 3 warmup and 20 timed iterations with auto-scaled iteration count.
Baseline is OpenBLAS via NumPy 2.4.3 (not MKL).

| Size      | Forge GF/s | NumPy GF/s | Ratio | Forge vs 192 GF/s peak |
| --------- | ---------- | ---------- | ----- | ---------------------- |
| 256×256   | 119.5      | 165.5      | 0.72× | 62%                    |
| 512×512   | 163.8      | 184.4      | 0.89× | 85%                    |
| 1024×1024 | 194.9      | 189.1      | 1.03× | 101%                   |

The 192 GF/s peak is computed as 2 cores × 3.0 GHz × 16 FP32 lanes (AVX-512) × 2 FLOP/FMA.
The 1024×1024 result at 101% efficiency reflects measured throughput against the
nominal base-clock model with turbo disabled. The 256 and 512 gaps are dominated
by packing and launch overhead, not arithmetic throughput — the kernel is efficient
once those costs amortize.

## Architecture

### `forge-lang`

`forge-lang` is a hand-written lexer plus Pratt parser with panic-mode recovery and explicit source spans. The parser interns symbols and resolves names with a no-shadowing rule, which is stricter than many language frontends but simplifies downstream SSA formation and diagnostics. Builtin paths like `thread::x` and `forge::fma` are recognized in the parser and resolver, so lowering can map them directly to IR intrinsics.

### `forge-ir`

`forge-ir` uses typed SSA with block parameters and edge arguments instead of phi nodes. This choice makes value flow explicit on CFG edges and lets the verifier check arity and type consistency per edge, rather than reconstructing phi semantics after the fact. `SsaBuilder` uses a sealed-block algorithm with incomplete parameters, which avoids a separate phi-insertion repair pass and keeps loop/branch lowering local.

### `forge-opt`

`forge-opt` runs a fixed-point pass manager with a 20-iteration cap. The key design tradeoff is simplicity and composability over aggressive one-pass sophistication: each pass is narrow, and convergence comes from repetition. Divergence analysis is placed in-pipeline before vectorization so widening only happens for loop/control regions that are uniform enough to be profitable.

### `forge-codegen`

`forge-codegen` emits Rust source with AVX intrinsics and lowers CFG control flow into a block-state loop. Instruction scheduling is list-based with a simple port-class heuristic, and register allocation is linear-scan over YMM locations; linear-scan was chosen over graph coloring because kernel-sized IR favors predictable compile cost over globally optimal coloring. A `rustc` validation gate exists as `compile_generated_source` and is exercised in tests, which gives a strong correctness backstop for emitter changes.

### `forge-runtime`

`forge-runtime` maps grid and block launch semantics to CPU threads using Rayon and crossbeam deques. `TensorAllocator` uses slab size classes with 64-byte alignment so packed tensor data is naturally aligned for wide SIMD loads and avoids repeated allocator jitter on hot paths. `WarpMask` and branch-mask helpers model lane predicates explicitly, which makes divergence semantics testable without relying on hardware warp behavior.

### `forge-std`

`forge-std` contains reference kernels, with matmul as the most tuned path and relu/softmax/layer_norm/conv2d as correctness-oriented kernels. The matmul implementation uses BLIS-style panel packing and architecture-specific microkernels, while keeping scalar and edge-tile fallbacks for correctness across non-multiple shapes. The library is both a runtime baseline and a tuning surface for the compiler backend.

### `forge-driver`

`forge-driver` is the orchestration layer for compile, check, dump, and benchmark workflows. It gives full stage visibility through `dump` and ties together lexer, parser, resolver, lowering, optimization, and codegen in one CLI path. The `run` command is currently a scaffold that reports launch configuration and compiled output path rather than executing generated kernels end-to-end.

## The Matmul Kernel

The matmul path is a blocked GEMM in the BLIS family: pack `A` into `MR x KC` panels, pack `B` into `KC x NR` panels, and drive a microkernel over packed panels. The primary register tiles are `MR=6` with `NR=16` on AVX2 and `NR=32` on AVX-512, with dedicated small-kernel tiles `MR_SMALL=4`, `NR_SMALL_AVX2=8`, and `NR_SMALL_AVX512=16` for low-dimension cases.

The kernel dispatch is three-tiered. A direct small path is enabled only for very small cubic problems up to `128^3`, a packed small/medium path covers dimensions up to 192 and 512 with tuned blocking, and a general packed path handles larger matrices. The code also downshifts from AVX-512 to AVX2 below an explicit flop threshold (`AVX512_DOWNSHIFT_FLOPS = 192^3`) to avoid paying AVX-512 setup and frequency penalties on tiny workloads.

The K-loop in both AVX2 and AVX-512 microkernels is unrolled by 4 (`kc_main = kc & !3` with four fused k-steps). On Tiger Lake this is a latency-hiding move: each step creates independent FMA accumulation chains while prefetching ahead (`PREFETCH_K_DIST = 8`), so broadcast, load, and FMA latency overlap instead of serializing on one chain.

The current implementation does not hard-code `MC=48, KC=128`. A simple 48 KB L1 arithmetic model suggests that pair as a conservative starting point, but this codebase uses cache-probed autotuning with dynamic MC balancing and KC values that depend on path and cache size (`KC` often 160 to 256 in practice, with medium-path overrides). That is a deliberate deviation driven by measured throughput and thread-balance behavior rather than fixed textbook tiles.

Inference from the current thresholds and comments: the small direct fast path exposed a regression window on Tiger Lake when strided direct accesses grew beyond tiny problems, while packed panels remained stable. The likely mechanism is prefetch behavior: Tiger Lake prefetchers reward regular contiguous panel streams and penalize odd-width and odd-K access patterns, which aligns with the explicit edge-aware KC reduction (`EDGE_AWARE_KC=160`) for irregular shapes.

## Optimization Pipeline

The pass manager iterates until convergence or 20 iterations, so transformations can enable each other without manual phase restarts. The core optimization sequence is eight transformation passes, with divergence analysis feeding vectorization in between.

| Order | Pass                             | What it does                                                | What it enables next                               |
| ----: | -------------------------------- | ----------------------------------------------------------- | -------------------------------------------------- |
|     1 | `SimplifyCfgPass`                | Prunes unreachable blocks and merges trivial jump chains    | Reduces CFG noise before expression-level rewrites |
|     2 | `ConstantFolding`                | Replaces foldable arithmetic and comparisons with constants | Exposes identity patterns and dead defs            |
|     3 | `StrengthReductionPass`          | Converts expensive integer ops to shifts and identities     | Creates dead defs and canonical forms for DCE/CSE  |
|     4 | `DeadCodeElimination`            | Removes side-effect-free unused defs                        | Shrinks IR before global expression dedup          |
|     5 | `CommonSubexpressionElimination` | Reuses dominating equivalent expressions                    | Normalizes producer graph before fusion            |
|     6 | `FmaFusionPass`                  | Fuses `FMul + FAdd` into `FMA`                              | Gives vectorizer and backend explicit FMA nodes    |
|     7 | `VectorizationPass`              | Widens eligible counted loops and patches induction step    | Produces vector IR patterns with alignment demands |
|     8 | `MemoryLayoutOptimization`       | Propagates stronger alignment on load/store pointers        | Improves final instruction selection quality       |

A final cleanup DCE is run again after vectorization and memory-layout rewrites. Divergence analysis is intentionally run in the pipeline even though it is not a mutating pass, because vectorization decisions depend on fresh uniform versus divergent classification.

## Getting Started

Build and test from the workspace root with:

```bash
cargo build --release -p forge-driver
cargo test --workspace
```

Compile, inspect, and benchmark with:

```bash
cargo run --release -p forge-driver -- check examples/matmul.forge
cargo run --release -p forge-driver -- dump examples/matmul.forge --stage ir-opt
cargo run --release -p forge-driver -- compile examples/matmul.forge
cargo run --release -p forge-driver -- bench --size 1024 --warmup 5 --iters 20
```

For stable benchmark runs on this project, use WSL2 or Linux and keep thread controls explicit. The repository scripts already set `FORGE_RAYON_THREADS` and BLAS thread variables for this reason.

If you need repeatable frequency behavior on Intel CPUs, disable turbo during measurement and restore it afterward:

```bash
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
# run benchmark suite
echo 0 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
```

## Roadmap

| Area                      | Status           | What works now                                                                                                         | Limitation in current code                                                                                                                  | Next concrete step                                                   |
| ------------------------- | ---------------- | ---------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| Frontend and SSA lowering | Working          | Lexer, Pratt parser, resolver, sealed SSA lowering, verifier-backed IR pipeline                                        | Type lowering still maps some named types conservatively                                                                                    | Add richer type mapping and stricter lowering diagnostics            |
| Optimizer stack           | Working          | Fixed-point pass manager with CFG simplification, folding, reduction, CSE, FMA fusion, vectorization, alignment tuning | Pass interactions are simple and mostly local, no advanced alias analysis                                                                   | Add alias-aware memory reasoning and stronger loop dependence checks |
| AVX backend emission      | Working          | AVX2 and AVX-512 microkernel paths, block-state code emission, list scheduling                                         | Linear-scan allocator is implemented but not integrated into final emission path                                                            | Wire allocation decisions into emission and spill materialization    |
| Validation gate           | Partial          | `compile_generated_source` exists and is tested                                                                        | Driver compile path does not invoke gate by default                                                                                         | Run the rustc gate in `forge-driver compile` with an opt-out flag    |
| Runtime launch and memory | Working          | Rayon launch model, crossbeam work stealing, barrier, 64-byte tensor allocator                                         | Block execution is modeled on CPU threads, not a true device scheduler                                                                      | Add NUMA-aware allocation and launch pinning controls                |
| End-to-end execution CLI  | Partial          | `compile`, `check`, `dump`, and benchmark paths are functional                                                         | Dynamic loading and invocation of generated kernels is not yet wired; compilation, inspection, and benchmarking are fully functional today. | Implement dynamic loading and invocation of generated kernels        |
| Kernel library            | Working baseline | Matmul tuned path plus relu, softmax, layer_norm, exp, conv2d                                                          | Non-matmul kernels prioritize correctness over heavy tuning                                                                                 | Add per-kernel benchmark gates and tuning loops like matmul          |

## Design Notes

### Sealed SSA and block arguments

The lowering path uses sealed SSA construction so variables are resolved as soon as predecessor information is complete. This avoids global phi-placement algorithms and makes variable merge points visible in CFG edges. The verifier can then enforce edge-argument arity directly, which simplifies debugging malformed control flow.

### CFG as executable state machine

Generated kernels are emitted as a `bb` state variable plus a loop and match over block IDs. This mirrors IR control flow directly and keeps block-parameter moves explicit at jump sites. The result is verbose source, but each SSA edge remains inspectable and reproducible after code generation.

### Alignment contract across layers

`TensorAllocator` guarantees 64-byte alignment, `SharedMemory` uses 32-byte alignment for AVX2 compatibility, and the matmul path propagates alignment assumptions through packed panels and memory-layout tuning. That cross-layer contract is why load/store alignment can be optimized late without unsafe alias speculation in mid-level passes.

## Why Forge exists

Forge exists because CPU execution is still the default deployment target for a large amount of inference and systems code, but most kernel compiler investment is GPU-first. This project is an attempt to make the CPU path first-class: transparent IR, explicit optimization tradeoffs, inspectable generated code, and a runtime model that can be debugged with ordinary host tooling. The goal is not to imitate GPU stacks on CPU. The goal is to make a compiler that takes CPU constraints seriously enough to be useful as both infrastructure and research software.
