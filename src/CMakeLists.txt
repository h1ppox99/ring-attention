# Ring attention CUDA/C++ library sources will go here.
# Example:
#   add_library(ring_attention STATIC ring_attention.cu ring_attention.hpp)
#   target_include_directories(ring_attention PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
#   target_link_libraries(ring_attention PUBLIC MPI::MPI_CXX CUDA::cudart)

add_library(ring_attention STATIC
  device_tensor.cu
  cpu_attention.cpp
  naive_attention.cu
  flash_attention.cu
  attention_step.cu
  ring_partition.cpp
  ring_loop.cu
)

target_include_directories(ring_attention PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ring_attention PUBLIC CUDA::cudart MPI::MPI_CXX)
set_target_properties(ring_attention PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
)
