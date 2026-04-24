cmake_minimum_required(VERSION 3.24)
project(ring_attention LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  # For clangd, clang-tidy

# Detect target GPU architecture; override via -DCMAKE_CUDA_ARCHITECTURES=...
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)  # Turing (T4/RTX 2080)
endif()

find_package(MPI REQUIRED)
find_package(CUDAToolkit REQUIRED)

# ccache if available
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
  set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()

enable_testing()

add_subdirectory(src)
add_subdirectory(apps)
add_subdirectory(tests)
