add_executable(test_smoke test_smoke.cu)
target_link_libraries(test_smoke PRIVATE ring_attention)
add_test(NAME smoke COMMAND test_smoke)
