cmake_minimum_required(VERSION 3.16)
project(optimized_cpu_gemm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags
add_compile_options(-O3 -Wall -Wextra -march=native)

# Executable
add_executable(gemm_benchmark
  src/main.cpp
  src/gemm_naive.cpp
  src/gemm_blocked.cpp
  src/gemm_simd.cpp
)
