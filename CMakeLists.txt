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

# ---------------------------------------------------------------------------
# Optional NCCL back-end for direct GPU-to-GPU ring communication.
# Enabled by default when the NVHPC module is loaded; disable with
#   cmake --preset=release -DUSE_NCCL=OFF
# ---------------------------------------------------------------------------
option(USE_NCCL "Use NCCL for GPU-to-GPU ring communication (eliminates host staging)" ON)
if(USE_NCCL)
  # The NVHPC 24.1 module (activate.sh) prepends the NCCL directories to
  # CPATH and LD_LIBRARY_PATH, so find_path / find_library pick them up via
  # those hints.  Override with -DNCCL_ROOT=<path> if the module is not loaded.
  set(NCCL_ROOT "" CACHE PATH "Root directory of the NCCL installation (optional override)")
  # CPATH and LD_LIBRARY_PATH are colon-separated; convert to CMake semicolon lists.
  string(REPLACE ":" ";" _cpath_list "$ENV{CPATH}")
  string(REPLACE ":" ";" _ldpath_list "$ENV{LD_LIBRARY_PATH}")
  find_path(NCCL_INCLUDE_DIR nccl.h
            HINTS "${NCCL_ROOT}/include" ${_cpath_list}
            REQUIRED)
  find_library(NCCL_LIBRARY nccl
               HINTS "${NCCL_ROOT}/lib" ${_ldpath_list}
               REQUIRED)
  message(STATUS "NCCL include : ${NCCL_INCLUDE_DIR}")
  message(STATUS "NCCL library : ${NCCL_LIBRARY}")
endif()

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
