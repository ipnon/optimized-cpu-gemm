# Month 2: High-Performance CPU Programming

## Objectives

- Understand caches, NUMA, SIMD intrinsics, and blocking strategies.
- Learn Linux performance profiling tools (perf, flamegraphs).

## Readings

- Systems Performance (Gregg): CPU, Memory, Profiling chapters.
- Intel Intrinsics Guide.
- Brendan Gregg: "Flamegraphs."

## Project: Optimized CPU GEMM (Matrix Multiply)

### Requirements

- Implement three versions:
  1. Naive triple-for-loop (baseline)
  2. Cache-blocked version (L1/L2 aware)
  3. SIMD-accelerated version using AVX2 or AVX-512
- Benchmark each on matrices 256×256, 512×512, 1024×1024.
- Use perf to collect:
  - Cache misses
  - Branch misses
  - Cycles-per-instruction (CPI)
- Generate a flamegraph for each version.
- Provide a performance report + graphs.

### Why This Matters

- GEMM is the foundation of AI compute.
- Understanding CPU performance sets the stage for GPU optimization.
- Demonstrates mastery of low-level optimization → great for interviews.

### Notes

- NUMA is Non-Uniform Memory Access.
- SIMD is Single Instruction, Multiple Data.
- Intrinsics are C functions that map directly to SIMD instructions.
- Blocking is also called tiling. It divides large matrix operations into smaller chunks that fit into cache. L1 cache is ~100x faster than RAM. Every time the working set exceeds cache size, the cache misses and the CPU stalls.
- GEMM stands for General Matrix Multiply. 90% of transformer FLOPs are matrix multiplies.
- Cache-blocking means choosing tile sizes so that the tiles fit in either L1 or L2. On an Apple M2 CPU L1 is 128KB/core in ~4 cycles and L2 is 4MB/core in ~12 cycles.
- AVX2, AVX-512, and ARM NEON are all SIMD instruction sets, but M2 only uses NEON. They function identically but their intrinsic functions have different names.
- Branch prediction is relevant to GEMM because an ideal GEMM has zero branch misses.
- CPI is cycles per instruction. CPI ~= 1 is generally par. CPI <= 0.25 is excellent. CPI > 2 is abysmmal.
- `#pragma once` prevents headers from being included multiple times.
- We don't use nested `std::vector<float>` for matrices because they'd be scattered in memory. Instead we use raw pointers to device memory: `C[i * N + j] += A[i * K + k] * B[k * N + j]`.
- CUDA uses row-major matrix access by convention. Sequential memory access allows fast cache hits because CPUs load memory in cache lines (128 bytes for M2).
