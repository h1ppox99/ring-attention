# Ring-attention CUDA/C++ library: kernels, host orchestrators, and the CPU
# reference. Headers (e.g. ring_loop_common.hpp) are pulled in via #include and
# need not be listed. NCCL wiring is appended below, gated on USE_NCCL.

add_library(ring_attention STATIC
  device_tensor.cu
  cpu_attention.cpp
  naive_attention.cu
  flash_attention.cu
  flash_attention_fp16.cu
  attention_step.cu
  attention_step_fp16.cu
  attention_decode.cu
  ring_partition.cpp
  ring2d_schedule.cpp
  ring_loop.cu
  ring_loop_fp16.cu
  kv_cache.cu
  decode_op.cu
  ring_decode.cu
)

target_include_directories(ring_attention PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ring_attention PUBLIC CUDA::cudart MPI::MPI_CXX)

if(USE_NCCL)
  target_include_directories(ring_attention PRIVATE ${NCCL_INCLUDE_DIR})
  target_link_libraries(ring_attention PUBLIC ${NCCL_LIBRARY})
  target_compile_definitions(ring_attention PUBLIC RING_USE_NCCL)
  message(STATUS "ring_attention: NCCL back-end ENABLED")
else()
  message(STATUS "ring_attention: NCCL back-end DISABLED (MPI host-staging path)")
endif()
set_target_properties(ring_attention PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
)
