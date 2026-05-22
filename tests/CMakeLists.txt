add_executable(test_smoke test_smoke.cu)
target_link_libraries(test_smoke PRIVATE ring_attention)
add_test(NAME smoke COMMAND test_smoke)

add_executable(test_device_tensor test_device_tensor.cu)
target_link_libraries(test_device_tensor PRIVATE ring_attention)
add_test(NAME device_tensor COMMAND test_device_tensor)

add_executable(test_cpu_attention test_cpu_attention.cpp)
target_link_libraries(test_cpu_attention PRIVATE ring_attention)
add_test(NAME cpu_attention COMMAND test_cpu_attention)

add_executable(test_naive_attention test_naive_attention.cu)
target_link_libraries(test_naive_attention PRIVATE ring_attention)
add_test(NAME naive_attention COMMAND test_naive_attention)

add_executable(test_flash_attention test_flash_attention.cu)
target_link_libraries(test_flash_attention PRIVATE ring_attention)
add_test(NAME flash_attention COMMAND test_flash_attention)

add_executable(test_flash_attention_fp16 test_flash_attention_fp16.cu)
target_link_libraries(test_flash_attention_fp16 PRIVATE ring_attention)
add_test(NAME flash_attention_fp16 COMMAND test_flash_attention_fp16)

add_executable(test_attention_step test_attention_step.cu)
target_link_libraries(test_attention_step PRIVATE ring_attention)
add_test(NAME attention_step COMMAND test_attention_step)

add_executable(test_ring_partition test_ring_partition.cpp)
target_link_libraries(test_ring_partition PRIVATE ring_attention)
add_test(NAME ring_partition COMMAND test_ring_partition)

add_executable(test_gen_elem test_gen_elem.cpp)
target_link_libraries(test_gen_elem PRIVATE ring_attention)
add_test(NAME gen_elem COMMAND test_gen_elem)

# Multi-rank smoke tests for the ring driver. Require 2 GPUs visible to
# mpirun, so they only pass from within `salloc --gres=gpu:2 --ntasks=2`.
# On the login node they fail the same way other GPU-tests do; that's expected.
# The matrix (causal + zigzag, all three modes) is the smallest configuration
# that catches a broken ring rotation or a regression in the sub-group loop.
find_program(MPIRUN_EXECUTABLE NAMES mpirun)
if(MPIRUN_EXECUTABLE)
  # Per-mode zigzag flag: allgather rejects --zigzag (it materializes the
  # full K/V, so token reordering is meaningless), the ring modes require it
  # to be the load-balanced configuration we actually ship.
  foreach(_mode allgather ring-blocking ring-overlap)
    if(_mode STREQUAL "allgather")
      set(_zigzag_flag "--zigzag" "0")
    else()
      set(_zigzag_flag "--zigzag" "1")
    endif()
    add_test(
      NAME ring_smoke_${_mode}
      COMMAND ${MPIRUN_EXECUTABLE} -n 2 $<TARGET_FILE:ring_attention_cli>
              --seq 128 --heads 2 --head_dim 32 --iters 1
              --causal 1 ${_zigzag_flag} --mode ${_mode} --verify)
    set_tests_properties(ring_smoke_${_mode} PROPERTIES
      PASS_REGULAR_EXPRESSION "verify[^\n]*PASS"
      LABELS "mpi;gpu")
  endforeach()
endif()
