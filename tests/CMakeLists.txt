add_executable(test_smoke test_smoke.cu)
target_link_libraries(test_smoke PRIVATE ring_attention)
add_test(NAME smoke COMMAND test_smoke)

add_executable(test_cpu_attention test_cpu_attention.cpp)
target_link_libraries(test_cpu_attention PRIVATE ring_attention)
add_test(NAME cpu_attention COMMAND test_cpu_attention)
