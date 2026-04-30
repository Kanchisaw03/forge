# Forge: A CPU-Native Kernel Compiler with SSA IR, AVX2 Emission, and Software Warp Semantics

**Forge** is a seven-crate Rust workspace that compiles a kernel language through typed SSA intermediate representation, a multi-pass fixed-point optimizer, and a register-allocated AVX2 backend — without touching a GPU, a GPU driver, or a GPU runtime. The generated artifact is verified Rust source validated by `rustc` before it leaves the compiler.

> *"The best GPU compiler is the one you can debug at 2am without a $50,000 accelerator."*

[![build](https://img.shields.io/badge/build-passing-brightgreen)](.) [![license](https://img.shields.io/badge/license-MIT-blue)](.) [![rust](https://img.shields.io/badge/rust-stable-orange)](.) [![crates](https://img.shields.io/badge/crates-7-lightgrey)](.)

---

## Table of Contents

1. [Motivation and Problem Statement](#1-motivation-and-problem-statement)
2. [Design Philosophy](#2-design-philosophy)
3. [Architecture](#3-architecture)
4. [Crate Reference](#4-crate-reference)
5. [Optimizer Pass Analysis](#5-optimizer-pass-analysis)
6. [Memory Model](#6-memory-model)
7. [Codegen and Verification Strategy](#7-codegen-and-verification-strategy)
8. [Software Warp Model](#8-software-warp-model)
9. [Comparison with Existing Compilers](#9-comparison-with-existing-compilers)
10. [Getting Started](#10-getting-started)
11. [Python Integration](#11-python-integration)
12. [Benchmark Harness](#12-benchmark-harness)
13. [Roadmap](#13-roadmap)
14. [Design Decisions and Trade-offs](#14-design-decisions-and-trade-offs)
15. [Why Rust](#15-why-rust)
16. [Contributing](#16-contributing)

---

## 1. Motivation and Problem Statement

The dominant GPU compiler stacks — CUDA, ROCm/HIP, Triton, XLA — share a structural assumption: that the target is a massively parallel GPU with device memory, streaming multiprocessors, and a proprietary driver layer. That assumption is load-bearing throughout their design. It shapes memory models, execution semantics, debugging tools, and deployment requirements.

For a large class of production workloads — batch inference on CPU servers, edge deployment, latency-sensitive inference without GPU provisioning, or compiler research itself — that assumption is a liability, not an asset. The result is one of three outcomes: teams port their kernels to a CPU backend as an afterthought (losing optimization quality), they maintain two separate codebases (doubling maintenance cost), or they accept GPU-only deployment as a constraint (limiting where software can run).

**Forge attacks this from the compiler side.** Rather than treating CPU as a fallback target in a GPU-first stack, Forge makes CPU the primary target and builds the entire pipeline — language, IR, optimizer, codegen, runtime — around that decision. The output is not an interpreted representation or a JIT-compiled blob; it is verified, human-readable Rust source using `std::arch::x86_64` intrinsics, validated by `rustc`, and free of any GPU dependency at link time or runtime.

This matters for three reasons:

**Debuggability.** When the output is Rust source, you can read it, modify it, instrument it, and understand exactly what the compiler decided. Black-box PTX or device bitcode offers none of that.

**Deployment simplicity.** A binary compiled from Forge output runs on any x86_64 CPU with AVX2. No driver installation, no CUDA version matrix, no GPU availability check, no runtime initialization.

**Compiler research accessibility.** The entire pipeline from parse tree to emitted intrinsics is Rust code you can fork, modify, and instrument. Studying Forge teaches every stage of a real compiler; studying CUDA teaches you how to use a compiler.

---

## 2. Design Philosophy

Four principles govern every architectural decision in Forge:

**Correctness before performance.** The IR verifier runs after every lowering step. The codegen validator invokes `rustc` on every emitted file. Optimization passes carry explicit preconditions. A compiler that produces fast wrong code is worse than one that produces slow correct code.

**Layers over monoliths.** Each crate has exactly one job and a defined interface to its neighbors. `forge-lang` does not know what SSA is. `forge-ir` does not know what AVX2 is. `forge-codegen` does not know what a parse tree looks like. This separation makes each layer independently testable and replaceable.

**Observability as a first-class feature.** `check`, `dump`, and `info` are not debug commands added after the fact — they are primary user-facing operations. A compiler that cannot show you its intermediate state is a compiler you cannot trust.

**The CPU execution model is not a limitation — it is a design constraint that produces better software.** Constraining the target to one ISA (x86_64 AVX2), one threading model (Rayon + crossbeam work stealing), and one memory space (host DRAM) eliminates an enormous class of bugs that GPU-targeting compilers must handle: device/host memory consistency, warp divergence on real hardware, driver version sensitivity, and non-deterministic kernel scheduling.

---

## 3. Architecture

```
                                    ┌─────────────────────────────┐
                                    │        forge-driver          │
                                    │  compile / run / bench       │
                                    │  check / dump / info         │
                                    │  [feature: python] PyO3 API  │
                                    └──────────────┬──────────────┘
                                                   │
                    ┌──────────────────────────────▼──────────────────────────────┐
                    │                    Compilation Pipeline                      │
                    │                                                              │
    .forge source   │   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌────────┐  │
    ───────────────►│──►│forge-lang│──►│ forge-ir │──►│forge-opt │──►│forge-  │  │
                    │   │          │   │          │   │          │   │codegen │  │
                    │   │ Lexer    │   │ SSA IR   │   │ 8 passes │   │        │  │
                    │   │ Pratt    │   │ Lowering │   │ Fixed pt │   │ AVX2   │  │
                    │   │ Parser   │   │ Verifier │   │ 20 iters │   │ Sched  │  │
                    │   │ NameRes  │   │          │   │          │   │ RegAlc │  │
                    │   └──────────┘   └──────────┘   └──────────┘   └───┬────┘  │
                    └──────────────────────────────────────────────────── │ ──────┘
                                                                           │
                                                              Rust source + intrinsics
                                                                           │
                                                                     ┌────▼────┐
                                                                     │  rustc  │
                                                                     │validate │
                                                                     └────┬────┘
                                                                           │
                                                              native AVX2 binary
                                                                           │
                    ┌──────────────────────────────────────────────────────▼─────┐
                    │                     Execution Layer                         │
                    │                                                              │
                    │   ┌──────────────────────────┐   ┌────────────────────┐    │
                    │   │      forge-runtime        │   │     forge-std      │    │
                    │   │                          │   │                    │    │
                    │   │  launch_2d (Rayon tasks) │   │  matmul (32x32x32) │    │
                    │   │  work-stealing deque     │   │  relu / softmax    │    │
                    │   │  WarpMask (8-lane)       │   │  layer_norm        │    │
                    │   │  TensorAllocator (slab)  │   │  exp_approx / fma  │    │
                    │   │  barrier / sync          │   │  conv2d            │    │
                    │   └──────────────────────────┘   └────────────────────┘    │
                    └──────────────────────────────────────────────────────────────┘
```

Data flows in one direction: source text enters `forge-lang`, typed SSA exits `forge-codegen` as Rust source, `rustc` validates and compiles it, and `forge-runtime` provides the CPU execution primitives. `forge-std` sits above the runtime and provides pre-tuned kernel implementations as a correctness baseline and performance reference.

---

## 4. Crate Reference

### `forge-lang` — Frontend

The frontend converts source text into a name-resolved AST ready for IR lowering.

**Lexer.** A hand-written lexer tokenizes the input using a character-class dispatch table. Keywords, operators, and identifiers are classified in a single pass. Source positions are tracked as byte offsets into the original input buffer for accurate diagnostic spans.

**Parser.** The parser is a Pratt parser — a top-down operator-precedence parser where each token carries a binding power that determines how tightly it binds as a prefix or infix operator. Pratt parsers handle expression grammars cleanly without the left-recursion gymnastics required by LL parsers and without the table-generation complexity of LALR. The grammar recognizes `@kernel fn` declarations and a set of built-in names: `thread::x`, `thread::y`, `block::x`, `block::y`, `warp::lane_id`, and synchronization primitives `forge::sync_block`, `forge::sync_warp`, `forge::atomic_add`, and math built-ins `forge::exp`, `forge::sqrt`, `forge::fma`.

**Panic-mode recovery.** When the parser encounters a syntax error, it does not abort. It advances the token stream to the next synchronization point (typically a statement boundary or closing delimiter) and continues parsing. This allows the compiler to report multiple independent errors in a single invocation — a meaningful quality-of-life improvement over compilers that stop at the first error.

**Name resolution.** A scoped symbol table walks the AST after parsing and resolves every identifier to its declaration. Unresolved names and duplicate declarations are reported as errors with the source span of both the use and the conflicting definition. The resolved AST is what enters IR lowering.

**Arena allocation.** AST nodes are allocated in a `bumpalo` arena. The entire AST is freed in a single deallocation when the arena is dropped, avoiding the per-node overhead of individual `Box` allocations and the fragmentation that would result from thousands of small heap allocations across a typical parse.

---

### `forge-ir` — Typed SSA Intermediate Representation

`forge-ir` defines the compiler's central data structure: a typed, explicitly-parameterized SSA IR.

**Value and block handles.** Values and basic blocks are represented as typed integer handles into per-function arenas. Handle-based IR avoids pointer invalidation when the arena grows and makes serialization, cloning, and verification straightforward.

**SSA construction via block parameters.** Rather than using phi nodes (as in LLVM-style IR), `forge-ir` uses block parameters and edge arguments — the "functional SSA" or "continuation-passing style" representation used in MLton, Cranelift, and MLIR. Every basic block declares a list of typed parameters. Control-flow edges carry explicit argument lists that are matched positionally to the target block's parameters. This makes SSA form explicit and verifier-checkable: a malformed SSA program is one where an edge argument list has the wrong arity or type, which the verifier can detect without understanding dominance.

**Verifier.** The IR verifier checks: every value use is dominated by its definition; every block is reachable from the function entry; every instruction has well-typed operands; every control-flow edge has argument lists that match the target block's parameter arity and types. Verification runs after lowering and after each optimization pass that modifies the CFG. A verification failure is a compiler bug, not a user error.

**Type system.** The IR carries a minimal type system: scalar integers and floats of standard widths, pointer types, and fixed-width vector types for AVX2 lowering. Types are interned for pointer-equality comparison.

---

### `forge-opt` — Optimizer

The optimizer is a fixed-point function-pass pipeline. Each pass implements a common trait; the `PassManager` runs all registered passes in sequence and repeats the sequence until no pass reports a change, up to a maximum of 20 iterations. Fixed-point iteration handles cases where one pass creates opportunities for another: constant folding may expose dead code, which DCE removes, which may expose new constant-foldable expressions.

Eight passes are currently registered. Each is described in detail in [Section 5](#5-optimizer-pass-analysis).

---

### `forge-codegen` — AVX2 Backend

`forge-codegen` translates Forge IR functions into Rust source files containing `unsafe extern "C"` functions annotated with `#[target_feature(enable = "avx2,fma")]`.

**Instruction selection.** Each IR instruction maps to one or more Rust expressions using `std::arch::x86_64` intrinsics. Scalar arithmetic maps directly; vectorizable regions emit `_mm256_*` intrinsics. The instruction selector handles the mismatch between the IR's abstract type system and the concrete SIMD type requirements of the intrinsic API.

**List scheduler.** Before register allocation, a list scheduler reorders instructions within a basic block to improve ILP (instruction-level parallelism). The scheduler respects data dependences (use-def chains) and targets latency heuristics for AVX2 and FMA operations.

**Linear-scan register allocator.** Register allocation uses a linear-scan allocator over the YMM register file (YMM0–YMM15 on x86_64 with AVX2). Linear-scan runs in O(n) time in the number of live intervals, which is acceptable for the kernel-sized functions Forge targets. Spill handling generates stack slot allocations and load/store pairs around spilled intervals.

**`rustc` validation gate.** After emission, `forge-codegen` invokes `rustc` on the generated file. A compile error from `rustc` is a codegen bug — it means the emitted source is not valid Rust or that an intrinsic was used incorrectly. This gate catches codegen regressions at the point of code generation rather than at the point where the downstream user attempts to compile.

---

### `forge-runtime` — CPU Execution Primitives

**`launch_2d`.** Maps the grid/block execution model onto Rayon parallel tasks. Each logical block becomes a Rayon task; the thread pool handles scheduling across physical cores. This gives Forge kernel launches the same two-level parallelism structure (grid of blocks, block of threads) as GPU execution without requiring GPU hardware.

**Work stealing.** The work-stealing deque is backed by `crossbeam-deque`. When a thread exhausts its local task queue, it steals tasks from the back of a neighbor's deque. Work stealing produces good load balance for the variable-granularity workloads that arise when blocks have different amounts of masking or early exit.

**`WarpMask`.** A software warp is an 8-lane execution context with an explicit active-lane bitmask. Divergent control flow (where different lanes take different branches) is modeled by executing both sides and blending results with `_mm256_blendv_ps`-style masking. `WarpMask` tracks which lanes are active in the current control-flow context.

**`TensorAllocator`.** A pooled slab allocator with fixed size classes from 64 bytes to 1 MiB, all 32-byte aligned for AVX2 load/store alignment. Allocations above 1 MiB fall through to the system allocator with their layout recorded for correct deallocation. `Tensor` wraps a raw pointer, element type, shape metadata, and an allocator reference; its `Drop` implementation returns the allocation to the appropriate size-class pool. This eliminates per-kernel allocation overhead at the cost of some memory overhead from size-class rounding.

**Barrier and sync primitives.** `sync_block` and `sync_warp` are CPU memory fences using `std::sync::atomic::fence` with `SeqCst` ordering. On x86_64, these lower to `MFENCE` instructions, which provide the same happens-before guarantee as their GPU counterparts.

---

### `forge-std` — Standard Kernel Library

`forge-std` provides six kernels: `matmul`, `relu`, `softmax`, `layer_norm`, `exp_approx`, and `conv2d`. These serve two purposes: they are correctness-tested reference implementations that validate the runtime, and they provide the starting point for future kernel fusion and vectorization work.

**Matmul** is the most performance-tuned path. It uses a 32×32×32 tiling strategy that fits the L1/L2 cache hierarchy on typical server CPUs, a 6×8 AVX2/FMA microkernel for the inner loop, and a scalar fallback for edge tiles where the dimensions do not divide evenly. The microkernel accumulates six rows of eight elements in YMM registers, minimizing load/store traffic across iterations.

The remaining kernels (relu, softmax, layer_norm, exp_approx, conv2d) are correct, tested implementations using standard algorithms. They are not yet maximally tuned but provide a correctness baseline and will be the target of future vectorization and fusion passes.

---

### `forge-driver` — Compiler Driver and CLI

`forge-driver` is the operational entry point. It wires together the full compilation pipeline and exposes six commands:

| Command | Function |
|---------|----------|
| `compile <file>` | Run the full pipeline and emit generated Rust source |
| `check <file>` | Run parse, name resolution, and IR lowering; report errors without emitting |
| `dump <file> --stage <stage>` | Print the IR or AST at a named pipeline stage |
| `bench` | Run the built-in SGEMM benchmark and print latency and GFLOPS |
| `info` | Report active CPU SIMD features, detected ISA extensions, and compiler version |
| `run <file>` | Compile and execute (execution integration in progress) |

The optional `python` Cargo feature builds a PyO3 extension module that exposes `forge.compile()` for Python callers.

---

## 5. Optimizer Pass Analysis

The optimizer runs eight passes in a fixed-point loop. The loop terminates when no pass modifies the IR or after 20 iterations, whichever comes first.

### `ConstantFolding`

Evaluates expressions whose operands are all compile-time constants. Arithmetic on integer and float literals, boolean comparisons with known operands, and identity operations (`x * 1`, `x + 0`, `x ** 0`) are all replaced with their results. Constant folding creates opportunities for every other pass: dead branches can be eliminated, CSE can find more matching expressions, and the vectorizer can prove loop bounds at compile time.

**What it eliminates:** Runtime arithmetic on values that were always known at compile time. In kernel code, this most commonly affects index offset calculations where stride or padding dimensions are compile-time constants.

### `DeadCodeElimination`

Removes SSA values that have no live uses and produce no observable side effects. DCE operates in two phases: a liveness analysis that marks all transitively live values starting from outputs, barriers, and stores; and a sweep that removes all unmarked instructions. The sweep also removes unreachable basic blocks — blocks that have no predecessors after constant folding removes conditional branches.

**What it eliminates:** Intermediate values computed by the frontend that are used in one control-flow branch but not another, after that branch is resolved by constant folding. Also eliminates the residue of CSE (replaced uses leaving the original computation without users).

### `CommonSubexpressionElimination`

Identifies pairs of instructions that compute the same value — same opcode, same operands in the same order — where one dominates the other. The dominated copy is replaced with a use of the dominating value. CSE uses a value-numbering approach: instructions are hashed by opcode and operand value numbers, and hash collisions are checked for exact equality.

**Precondition:** Correctness requires that the dominating instruction is in a block that dominates all uses of the dominated instruction. This is checked using the dominator tree from `compute_domtree`.

**What it eliminates:** Repeated address calculations in tight loops (e.g., `row * stride + col` computed in both a read and a write), repeated type conversions, and redundant predicate evaluations.

### `DivergenceAnalysis`

Classifies each SSA value and each branch condition as *uniform* (same value on all active lanes in a warp) or *divergent* (potentially different across lanes). The analysis is a dataflow pass: values derived only from uniform inputs are uniform; values with any divergent input are divergent; branch conditions derived from thread index or lane ID are divergent.

**What it enables:** The vectorization pass requires loop bounds and induction variables to be uniform. Without divergence analysis, the vectorizer would have to conservatively refuse to vectorize any loop that mentions a thread index, even if the index is used only outside the loop body.

**What it eliminates:** Unnecessary scalar fallback paths in the vectorizer for loops where the bound is provably uniform.

### `VectorizationPass`

Widens counted loops with uniform bounds to 8-lane AVX2 vector operations. The pass identifies innermost loops where: (1) the trip count is uniform per divergence analysis; (2) the loop body contains no control flow; (3) dependency analysis confirms no loop-carried dependences prevent widening. When all conditions hold, the loop's induction variable is replaced with a vector of eight consecutive values (`_mm256_set_epi32`), scalar operations are widened to their vector equivalents, and the loop step is multiplied by eight.

**What it produces:** Eight scalar iterations collapsed into one vector iteration, reducing loop overhead and enabling the CPU's SIMD execution units.

**Current limitation:** The vectorizer handles innermost loops only. Outer loop vectorization and loop unroll-and-jam are not yet implemented.

### `compute_domtree`

Computes the dominator tree of the control-flow graph using the iterative dataflow algorithm. A block D dominates a block B if every path from the function entry to B passes through D. The dominator tree is used by CSE (to check the domination precondition), by DCE (to identify unreachable blocks), and by the verifier (to check SSA def-dominates-use).

This is not strictly an optimization pass — it is an analysis that other passes consume. It is registered in the pass pipeline so that its result is always fresh when downstream passes request it.

### `MemoryLayoutOptimization`

Analyzes pointer provenance to determine whether load and store instructions can be given stronger alignment annotations. When a pointer is provably derived from a `TensorAllocator` allocation (which guarantees 32-byte alignment), load and store instructions are annotated with `align(32)`, which allows the backend to emit aligned AVX2 load/store intrinsics (`_mm256_load_ps`) instead of unaligned variants (`_mm256_loadu_ps`). Aligned loads are faster on most microarchitectures because they avoid the cross-cache-line penalty.

**What it produces:** Aligned AVX2 memory operations where pointer provenance can be established statically.

### `PassManager`

Not a transformation pass, but the fixed-point driver. It runs all registered passes in registration order, tracks whether any pass reported a modification, and repeats the sequence if any modification occurred. The 20-iteration cap prevents pathological cases where two passes create and destroy the same pattern indefinitely (though this should not occur with the current set of passes).

---

## 6. Memory Model

Forge's memory model is intentionally CPU-native. There is no device memory, no pinned host memory, no DMA, and no page-table manipulation. All tensor data lives in ordinary host DRAM, accessed through 32-byte aligned pointers.

### TensorAllocator Design

```
Size classes (bytes): 64, 128, 256, 512, 1K, 2K, 4K, 8K, 16K, 32K, 64K, 128K, 256K, 512K, 1M
Alignment: 32 bytes (AVX2 load/store requirement)
Overflow: system allocator, layout recorded for deallocation
```

The allocator maintains one slab per size class. Each slab is a free list of fixed-size blocks. Allocation is O(1): pop the head of the appropriate free list, or extend the slab if the list is empty. Deallocation is O(1): push the block back onto the free list. The `Tensor::drop` implementation does this automatically.

Size classes are chosen to cover typical tensor shapes in the kernel library. A 256×256 `f32` tensor is 256 KiB, which falls in the 256K size class with no waste. A 32×32 tile is 4 KiB, exactly at the 4K boundary. The 32-byte alignment matches both the AVX2 register width and a typical cache line pair.

### Comparison with GPU Memory Models

| Property | Forge (CPU) | CUDA | Triton |
|----------|------------|------|--------|
| Address space | Single (host RAM) | Host + device | Device (with host copies) |
| Alignment | 32 bytes (explicit) | 128 bytes (device) | 128 bytes (device) |
| Pool specialization | Size classes | cudaMalloc / cudaMallocAsync | torch.empty |
| Fragmentation | Size-class rounding | Device heap fragmentation | PyTorch allocator |
| NUMA policy | None yet | N/A | N/A |
| Page management | OS-managed | Driver-managed | Driver-managed |
| Deterministic latency | Yes (free list O(1)) | cudaMallocAsync (approx) | No |

### Current Limitations and Roadmap

The current allocator does not implement NUMA-aware placement (allocating tensors on the NUMA node closest to the CPU cores that will process them), huge page backing (to reduce TLB pressure on large tensors), or static tensor lifetime analysis (to pre-size pools before the first kernel launch). These are the next three milestones for the memory subsystem.

---

## 7. Codegen and Verification Strategy

The codegen strategy is worth explaining in detail because it makes an unusual choice: it emits source code rather than object code.

Most compiler backends emit object code directly (LLVM IR → machine code via LLVM's backend, PTX → device binary via the CUDA driver). Forge emits Rust source that uses `std::arch::x86_64` intrinsics, then invokes `rustc` as a final compilation step.

This choice has three consequences:

**Human-readable output.** The generated Rust source is readable and inspectable. A developer debugging a miscompilation can look at exactly what the compiler emitted, not a disassembly of binary output.

**Correctness gate.** `rustc` enforces Rust's type system, borrow checker, and intrinsic API contracts on the generated code. A codegen bug that produces type-incorrect output or misuses an intrinsic API is caught at compile time, not at runtime. This is a stronger safety net than most compiler backends provide.

**Linkability.** Generated kernels are ordinary Rust `extern "C"` functions. They link directly into any Rust binary or can be called from C via FFI. No custom linker, no special loader, no driver initialization.

The trade-off is compilation time: invoking `rustc` on every generated file adds seconds to the compilation pipeline. For a kernel compiler targeting development and batch workloads, this is acceptable. For a JIT compiler targeting hot-path recompilation, it would not be.

---

## 8. Software Warp Model

GPU warps are groups of 32 threads that execute in lockstep on a single streaming multiprocessor. Divergent control flow — where threads in the same warp take different branches — is handled by the hardware through predication: both branches execute, and the results of the non-taken branch are discarded.

Forge models this in software using `WarpMask`: an 8-lane mask (matching the AVX2 register width) that tracks which lanes are currently active. When a divergent branch is encountered, Forge executes both branches with appropriate masks and blends the results.

```
WarpMask { active: u8 }
  - All lanes active:      0b11111111 (0xFF)
  - Even lanes active:     0b01010101 (0x55)
  - First 4 lanes active:  0b00001111 (0x0F)
```

This model has lower overhead than real GPU warp divergence handling because: (1) lanes are 8 rather than 32, reducing the cost of executing both sides; (2) the mask operations map directly to `_mm256_blendv_ps` and `_mm256_movemask_ps`, which are single-cycle operations on modern CPUs; and (3) `DivergenceAnalysis` identifies uniform branches, which can be executed as scalar conditionals without masking overhead.

The software warp model makes Forge kernels written for GPU-style execution semantics (with `warp::lane_id`, `sync_warp`, and masked operations) portable to CPU without changing the source language.

---

## 9. Comparison with Existing Compilers

| Property | Forge | CUDA/NVCC | Triton | XLA | MLIR |
|----------|-------|-----------|--------|-----|------|
| Primary target | CPU (AVX2) | NVIDIA GPU | GPU (PTX) | GPU/TPU | Multi-target |
| IR type | Typed SSA | PTX / LLVM IR | Triton IR | HLO | MLIR dialects |
| SSA representation | Block params | Phi nodes | Phi nodes | SSA (HLO) | Block args |
| Output | Rust source | PTX / CUBIN | PTX | XLA binary | LLVM IR |
| Output verifiable by | rustc | driver | ptxas | XLA | llc |
| Optimizer | 8-pass fixed-point | NVVM / LLVM | LLVM | XLA HLO | MLIR transforms |
| Vectorization | 8-lane AVX2 | GPU warp (32/64) | GPU warp | Vector ISA | Target-dependent |
| Runtime dependency | None | CUDA runtime | PyTorch | TF/JAX | None |
| Memory model | Host slab | Device + host | Device | Device | Host |
| Inspectable IR | Yes (`dump`) | Partial | No | No | Yes |
| License | MIT | Proprietary | Apache 2 | Apache 2 | Apache 2 |
| Written in | Rust | C++ | C++ | C++ | C++ |

**Key differentiators not captured by the table:**

Forge is the only compiler in this list where the full pipeline — lexer, parser, IR, optimizer, codegen, runtime — is implemented in a single memory-safe language with no unsafe code outside of explicitly marked SIMD boundaries.

Forge is the only compiler in this list that validates its own output by invoking a production compiler (rustc) as a correctness gate before the output leaves the compiler.

Forge exposes every intermediate representation through a first-class CLI command (`dump`), making it suitable as a teaching tool and a research platform in addition to a production compiler.

---

## 10. Getting Started

### Prerequisites

- Rust stable toolchain (1.70 or later recommended)
- x86_64 CPU with AVX2 support (check: `grep avx2 /proc/cpuinfo`)
- `rustc` in PATH (required for codegen validation)

### Build and Test

```bash
git clone https://github.com/your-org/forge
cd forge
cargo build
cargo test --workspace
```

For SIMD-optimized release builds:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

### Kernel Syntax

A minimal kernel using the syntax the parser currently accepts:

```forge
@kernel
fn matmul(a: &[f32], b: &[f32], out: &mut [f32], N: u32) {
    let row = thread::x();
    let col = thread::y();

    if row < N && col < N {
        let mut sum: f32 = 0.0;
        for k in 0u32..N {
            sum += a[row * N + k] * b[k * N + col];
        }
        out[row * N + col] = sum;
    }
}
```

Available built-ins in the kernel language:

| Built-in | Semantics |
|----------|-----------|
| `thread::x()`, `thread::y()` | Thread index within block (x and y dimensions) |
| `block::x()`, `block::y()` | Block index within grid |
| `warp::lane_id()` | Lane index within the software warp (0–7) |
| `forge::sync_block()` | Block-scope memory barrier (MFENCE) |
| `forge::sync_warp()` | Warp-scope memory barrier |
| `forge::atomic_add(ptr, val)` | Atomic fetch-and-add |
| `forge::exp(x)` | Single-precision exponential |
| `forge::sqrt(x)` | Single-precision square root |
| `forge::fma(a, b, c)` | Fused multiply-add |

### Running the Compiler

```bash
# Validate syntax and name resolution only
cargo run -p forge-driver -- check examples/matmul.forge

# Full compilation pipeline — emits generated Rust source
cargo run -p forge-driver -- compile examples/matmul.forge

# Inspect IR after optimization
cargo run -p forge-driver -- dump examples/matmul.forge --stage ir-opt

# Inspect IR before optimization
cargo run -p forge-driver -- dump examples/matmul.forge --stage ir-raw

# Report active CPU SIMD features
cargo run -p forge-driver -- info

# Run built-in SGEMM benchmark
cargo run --release -p forge-driver -- bench
```

---

## 11. Python Integration

The `python` feature builds a PyO3 extension module for Python callers who want to integrate Forge into a Python-based ML workflow without writing Rust.

Enable the feature:

```bash
cargo build --release --features python
```

Minimal usage:

```python
import forge

# Compile a .forge kernel file; returns a handle to the compiled output
kernel = forge.compile("examples/matmul.forge")

# Path to the generated Rust source file
print(kernel.path())
```

The Python API currently exposes compilation and source path access. Execution integration (loading and calling the compiled kernel from Python) is the next planned step for the Python wrapper.

---

## 12. Benchmark Harness

The benchmark harness in `forge-driver bench` measures a built-in 256×256 single-precision GEMM. The measurement protocol is:

- 5 warmup iterations (not measured) to bring data into cache and allow frequency scaling to stabilize
- 20 timed iterations
- Reported metrics: `latency_ms` (mean wall time per iteration) and `gflops` (derived from `2 * N^3` floating-point operations for an N×N GEMM)

**Important:** The repository does not yet include claimed performance numbers. The harness is in place and produces output, but results have not been validated against a tuned BLAS reference or published as baseline figures. The matmul microkernel (32×32×32 tiling, 6×8 AVX2/FMA inner loop) is the most optimized path; the other kernels in `forge-std` are correctness-correct but not yet performance-tuned.

Measured results will be added to this section once the execution integration in `forge-driver run` is complete and kernel outputs can be validated for correctness against reference implementations.

---

## 13. Roadmap

Forge's roadmap is organized by layer, from the execution layer (most immediately useful) to the memory layer (highest long-term impact).

### v0.2 — Close the Execution Loop
The `forge-driver run` command currently validates compile-time plumbing but does not execute compiled kernels end to end. The v0.2 milestone closes this loop: load the compiled kernel, wire it to the runtime launch infrastructure, execute on host data, and return results to the caller. This is the prerequisite for measured performance validation.

### v0.3 — Codegen Coverage and Spill Robustness
The AVX2 backend handles the fast paths for the kernel library but does not have full IR instruction coverage. The v0.3 milestone expands instruction selection to cover the complete IR instruction set and makes spill handling robust for kernels with high register pressure (the current linear-scan allocator handles spills but the spill code generation is not fully exercised).

### v0.4 — Optimizer Depth
The current optimizer passes are correct but conservative. The v0.4 milestone strengthens the proof coverage for CSE (handling more expression forms), DCE (handling more side-effect-free instruction categories), divergence propagation (handling loop-carried uniformity), and alignment inference (tracking alignment through arithmetic on aligned pointers).

### v0.5 — Memory System Maturity
Add NUMA-aware tensor placement, huge page backing for large allocations, and static tensor lifetime analysis that feeds pool sizing at compile time rather than at runtime. The static lifetime analysis is the most interesting piece: it requires the compiler to reason about tensor lifetimes across kernel boundaries, which is a cross-layer analysis touching `forge-ir`, `forge-opt`, and `forge-runtime`.

### v0.6 — Forge-std Benchmarked
Measure all six `forge-std` kernels against reference implementations (OpenBLAS for matmul, hand-written C intrinsics for the others), publish the results, and close the gap where it exists. This milestone makes Forge a credible performance story, not just a correctness story.

---

## 14. Design Decisions and Trade-offs

These are the decisions where reasonable engineers might disagree. They are documented here so that contributors understand why the code is the way it is.

**Block parameters instead of phi nodes.** Phi nodes are ubiquitous in LLVM-style IR; block parameters are used in Cranelift, MLIR, and functional language backends. Block parameters make SSA form syntactically explicit (the parameter is the definition, the edge argument is the use) and easier to verify (check arity and types on each edge). The trade-off is that some algorithms written for phi-node IR (particularly certain forms of SCCP and GVN) require adaptation. We chose correctness of the verifier over familiarity of the representation.

**Source-level codegen instead of object code.** Emitting Rust source and invoking `rustc` is slower than emitting object code directly. It is also more debuggable, more portable (works on any platform where `rustc` is available), and self-validating. For a kernel compiler where compilation happens ahead-of-time, the extra seconds are acceptable. This decision would need revisiting for a JIT scenario.

**AVX2 as the sole SIMD target.** AVX-512 offers 16-lane vectors and better masking, but AVX-512 is not universally available (it is absent on Apple Silicon entirely and on some Intel consumer CPUs). AVX2 is the common denominator for x86_64 server CPUs produced in the last decade. We target AVX2 first and will add AVX-512 and ARM NEON/SVE as secondary targets.

**8-lane software warps.** GPU warps are 32 or 64 lanes. We chose 8 lanes to match AVX2's `float` vector width (`_mm256_` holds 8 `f32` values). This makes the lane model and the SIMD model coincide exactly, simplifying codegen. A 32-lane warp model on AVX2 would require multi-register representations.

**Fixed-point optimizer with a 20-iteration cap.** A worklist-based optimizer (processing only the instructions affected by each change) would be faster. Fixed-point is simpler to implement correctly and to reason about. 20 iterations is enough for the current pass set to converge on any input we have tested. This is a performance engineering trade-off, not a correctness issue.

---

## 15. Why Rust

Rust is not a neutral choice for a compiler implementation. It is the right choice for Forge for reasons that are specific to what a compiler does.

A compiler manipulates unsafe data structures — arenas, raw pointers, intrusive linked lists, SIMD intrinsics, and foreign-language interface boundaries — in a context where a bug produces a miscompiled program rather than a crash. In C++, these structures are common but the type system does not enforce the ownership and aliasing rules that make them safe. In Rust, the borrow checker enforces those rules at compile time; `unsafe` blocks are explicit, auditable, and localized.

Forge's `unsafe` surface is small and well-contained: the SIMD intrinsics in `forge-codegen` and `forge-runtime`, the raw pointer arithmetic in `TensorAllocator`, and the FFI boundary in the PyO3 wrapper. Everything else — the parser, the IR, the optimizer passes, the verifier, the driver — is safe Rust. This means that a large class of compiler bugs (use-after-free in the arena, data races in the parallel runtime, out-of-bounds access in the register allocator) are impossible by construction, not merely tested against.

The zero-cost abstraction story matters for compiler performance. Iterators over IR instructions, pattern matching on instruction opcodes, and hash map lookups in the symbol table all compile to code as efficient as hand-written C. The abstraction cost is paid at compile time, not at runtime.

---

## 16. Contributing

Start by reading the architecture documentation in the design docs directory (parts 1–4), then review `FORGE_SESSION_CONTEXT.md` for the current implementation status of each layer.

The contribution workflow is:

```bash
cargo fmt                    # Format before committing
cargo test --workspace       # All tests must pass
cargo build --release        # Release build must succeed
RUSTFLAGS="-C target-cpu=native" cargo build --release   # SIMD build must succeed
```

Contribution priorities, in order:

1. **Correctness improvements** — additional verifier checks, new test cases for edge conditions, fuzz targets for the parser and IR
2. **Observability improvements** — better diagnostic messages, richer `dump` output, IR pretty-printer improvements
3. **Coverage improvements** — additional IR instruction support in codegen, additional kernel patterns in forge-std
4. **Performance improvements** — only after correctness and coverage are solid

Avoid placeholders (`todo!()`, `unimplemented!()`, `FIXME`). If something cannot be finished, discuss it in an issue rather than leaving a marker in the code. Keep changes to one layer at a time — a PR that modifies `forge-lang` and `forge-codegen` simultaneously is harder to review and easier to get wrong.

---

*Forge is a research-grade CPU compiler stack. It is not a production-ready tool yet. It is, however, a real compiler — every layer is implemented, the pipeline builds, tests pass, and the major components are wired together. The gap between "real compiler" and "production compiler" is engineering work, not architectural rethinking.*

---

**License:** MIT  
**Language:** Rust (stable)  
**Architecture:** x86_64 (AVX2 primary; AVX-512 and ARM on roadmap)  
**Dependencies:** bumpalo, rayon, crossbeam-deque, pyo3 (optional)