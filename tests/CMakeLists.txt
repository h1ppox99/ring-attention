add_executable(test_smoke test_smoke.cu)
target_link_libraries(test_smoke PRIVATE ring_attention)
add_test(NAME smoke COMMAND test_smoke)

add_executable(test_device_tensor test_device_tensor.cu)
target_link_libraries(test_device_tensor PRIVATE ring_attention)
add_test(NAME device_tensor COMMAND test_device_tensor)

add_executable(test_kv_cache test_kv_cache.cu)
target_link_libraries(test_kv_cache PRIVATE ring_attention)
add_test(NAME kv_cache COMMAND test_kv_cache)

add_executable(test_decode_op test_decode_op.cu)
target_link_libraries(test_decode_op PRIVATE ring_attention)
add_test(NAME decode_op COMMAND test_decode_op)

add_executable(test_ring_decode test_ring_decode.cu)
target_link_libraries(test_ring_decode PRIVATE ring_attention)
# Multi-rank smoke: prefill 32, decode 4 tokens across cp_size=2.
# Requires NCCL + 2 GPUs (same as ring_smoke_* tests). On a 1-GPU allocation
# this test will fail to bind both ranks; that's expected.
if(MPIRUN_EXECUTABLE)
  add_test(
    NAME ring_decode_smoke
    COMMAND ${MPIRUN_EXECUTABLE} -n 2 $<TARGET_FILE:test_ring_decode>
            --prompt_len 32 --n_decode 4 --batch 1 --heads 2 --head_dim 32)
  set_tests_properties(ring_decode_smoke PROPERTIES
    PASS_REGULAR_EXPRESSION "ring_decode[^\n]*PASS"
    LABELS "mpi;gpu")

  # Larger head_dim variant — exercises the kernel's D=64 dispatch path.
  add_test(
    NAME ring_decode_smoke_d64
    COMMAND ${MPIRUN_EXECUTABLE} -n 2 $<TARGET_FILE:test_ring_decode>
            --prompt_len 32 --n_decode 4 --batch 1 --heads 2 --head_dim 64)
  set_tests_properties(ring_decode_smoke_d64 PROPERTIES
    PASS_REGULAR_EXPRESSION "ring_decode[^\n]*PASS"
    LABELS "mpi;gpu")

  # 4-rank decode: exercises the cp_size=4 ring path. Needs 4 GPUs.
  add_test(
    NAME ring_decode_smoke_cp4
    COMMAND ${MPIRUN_EXECUTABLE} -n 4 $<TARGET_FILE:test_ring_decode>
            --prompt_len 32 --n_decode 4 --batch 1 --heads 2 --head_dim 32)
  set_tests_properties(ring_decode_smoke_cp4 PROPERTIES
    PASS_REGULAR_EXPRESSION "ring_decode[^\n]*PASS"
    LABELS "mpi;gpu")

  # CLI --run decode smoke: exercises the CLI's decode benchmark path
  # (run_decode_benchmark, --csv, --csv-header). Pass criterion: the summary
  # line printed on rank 0 ends with "cp_size=2".
  add_test(
    NAME ring_decode_cli_csv
    COMMAND ${MPIRUN_EXECUTABLE} -n 2 $<TARGET_FILE:ring_attention_cli>
            --run decode --prompt-len 64 --decode-tokens 3
            --batch 1 --heads 2 --head_dim 32 --csv-header --csv)
  set_tests_properties(ring_decode_cli_csv PROPERTIES
    PASS_REGULAR_EXPRESSION "decode summary[^\n]*cp_size=2"
    LABELS "mpi;gpu")
endif()

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

add_executable(test_attention_step_fp16 test_attention_step_fp16.cu)
target_link_libraries(test_attention_step_fp16 PRIVATE ring_attention)
add_test(NAME attention_step_fp16 COMMAND test_attention_step_fp16)

add_executable(test_ring_partition test_ring_partition.cpp)
target_link_libraries(test_ring_partition PRIVATE ring_attention)
add_test(NAME ring_partition COMMAND test_ring_partition)

add_executable(test_ring2d_schedule test_ring2d_schedule.cpp)
target_link_libraries(test_ring2d_schedule PRIVATE ring_attention)
add_test(NAME ring2d_schedule COMMAND test_ring2d_schedule)

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
  foreach(_mode allgather ring-blocking ring-overlap ring-2d)
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

    # FP16 / Tensor-Core path. Same smoke matrix, --dtype fp16. Tolerance
    # for the --verify check is the CLI's job; we just confirm the run
    # reaches "verify ... PASS" with the relaxed FP16 threshold.
    # ring-2d has no fp16 path yet, so skip its fp16 variant.
    if(NOT _mode STREQUAL "ring-2d")
      add_test(
        NAME ring_smoke_${_mode}_fp16
        COMMAND ${MPIRUN_EXECUTABLE} -n 2 $<TARGET_FILE:ring_attention_cli>
                --seq 128 --heads 2 --head_dim 32 --iters 1
                --causal 1 ${_zigzag_flag} --mode ${_mode} --dtype fp16 --verify)
      set_tests_properties(ring_smoke_${_mode}_fp16 PROPERTIES
        PASS_REGULAR_EXPRESSION "verify[^\n]*PASS"
        LABELS "mpi;gpu")
    endif()
  endforeach()

  # ring-2d, contiguous partition (nsg=1) — covers the non-zigzag sub-group path.
  add_test(
    NAME ring_smoke_ring-2d_contiguous
    COMMAND ${MPIRUN_EXECUTABLE} -n 2 $<TARGET_FILE:ring_attention_cli>
            --seq 128 --heads 2 --head_dim 32 --iters 1
            --causal 1 --zigzag 0 --mode ring-2d --verify)
  set_tests_properties(ring_smoke_ring-2d_contiguous PROPERTIES
    PASS_REGULAR_EXPRESSION "verify[^\n]*PASS"
    LABELS "mpi;gpu")

  # ring-2d on 4 ranks — a longer intra-node inner loop (G up to 4 on one node).
  # Needs 4 GPUs, like ring_decode_smoke_cp4.
  add_test(
    NAME ring_smoke_ring-2d_cp4
    COMMAND ${MPIRUN_EXECUTABLE} -n 4 $<TARGET_FILE:ring_attention_cli>
            --seq 128 --heads 2 --head_dim 32 --iters 1
            --causal 1 --zigzag 1 --mode ring-2d --verify)
  set_tests_properties(ring_smoke_ring-2d_cp4 PROPERTIES
    PASS_REGULAR_EXPRESSION "verify[^\n]*PASS"
    LABELS "mpi;gpu")
endif()
