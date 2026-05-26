/// @file
/// Unit tests for DeviceTensor: ctor/dtor, both copy_from_host / copy_to_host
/// overloads, move semantics, and all four template instantiations
/// (float, int, unsigned int, __half).
///
/// Exit code convention matches the other CUDA tests in this directory: print
/// "mismatch ..." to stderr and return 1 on failure, "device_tensor OK" + 0
/// on success.

#include <cuda_fp16.h>

#include <cstdint>
#include <cstdio>
#include <vector>

#include "device_tensor.hpp"

namespace {

int check(bool cond, const char* what) {
  if (!cond) {
    fprintf(stderr, "FAIL: %s\n", what);
    return 1;
  }
  return 0;
}

/// Round-trip via pointer overloads for POD types (float, int, unsigned).
template <typename T>
int test_roundtrip_ptr(const char* tag, const std::vector<T>& expected) {
  using namespace ring_attention;
  DeviceTensor<T> d(expected.size());
  if (check(d.size() == expected.size(), "size() after ctor")) return 1;
  if (check(d.bytes() == expected.size() * sizeof(T), "bytes() after ctor")) return 1;
  if (check(d.data() != nullptr, "data() non-null for non-empty tensor")) return 1;

  d.copy_from_host(expected.data());
  std::vector<T> got(expected.size());
  d.copy_to_host(got.data());
  for (std::size_t i = 0; i < expected.size(); ++i) {
    if (got[i] != expected[i]) {
      fprintf(stderr, "[%s] mismatch at %zu\n", tag, i);
      return 1;
    }
  }
  return 0;
}

/// Round-trip __half via pointer overloads. Comparison uses __half2float to
/// avoid relying on __half operator== in host code across compiler versions.
int test_roundtrip_half_ptr() {
  using namespace ring_attention;
  const float vals[] = {0.0f, 1.0f, -1.5f, 3.14f, 0.001f};
  constexpr int N = 5;

  std::vector<__half> expected(N);
  for (int i = 0; i < N; ++i) expected[i] = __float2half_rn(vals[i]);

  DeviceTensor<__half> d(N);
  if (check(d.size() == static_cast<std::size_t>(N), "half size() after ctor")) return 1;
  if (check(d.bytes() == static_cast<std::size_t>(N) * sizeof(__half), "half bytes()")) return 1;
  if (check(d.data() != nullptr, "half data() non-null")) return 1;

  d.copy_from_host(expected.data());
  std::vector<__half> got(N);
  d.copy_to_host(got.data());

  for (int i = 0; i < N; ++i) {
    if (__half2float(got[i]) != __half2float(expected[i])) {
      fprintf(stderr, "half ptr roundtrip mismatch at %d\n", i);
      return 1;
    }
  }
  return 0;
}

/// Round-trip via std::vector overloads. Exercises copy_from_host(vector) and
/// copy_to_host(vector) — the latter must resize the destination.
template <typename T>
int test_roundtrip_vector_overload(const char* tag, const std::vector<T>& expected) {
  using namespace ring_attention;
  DeviceTensor<T> d(expected.size());
  d.copy_from_host(expected);

  std::vector<T> got;  // intentionally empty; copy_to_host must resize it
  d.copy_to_host(got);
  if (check(got.size() == expected.size(), "vector overload resized destination")) return 1;
  for (std::size_t i = 0; i < expected.size(); ++i) {
    if (got[i] != expected[i]) {
      fprintf(stderr, "[%s] vector-overload mismatch at %zu\n", tag, i);
      return 1;
    }
  }
  return 0;
}

/// Round-trip __half via std::vector overloads.
int test_roundtrip_half_vector() {
  using namespace ring_attention;
  const float vals[] = {2.0f, -0.5f, 100.0f};
  constexpr int N = 3;

  std::vector<__half> expected(N);
  for (int i = 0; i < N; ++i) expected[i] = __float2half_rn(vals[i]);

  DeviceTensor<__half> d(N);
  d.copy_from_host(expected);

  std::vector<__half> got;
  d.copy_to_host(got);
  if (check(got.size() == static_cast<std::size_t>(N), "half vector overload size")) return 1;
  for (int i = 0; i < N; ++i) {
    if (__half2float(got[i]) != __half2float(expected[i])) {
      fprintf(stderr, "half vector-overload mismatch at %d\n", i);
      return 1;
    }
  }
  return 0;
}

/// Move constructor. Exercises DeviceTensor(DeviceTensor&&) for type T.
template <typename T>
int test_move_constructor(const char* tag, const std::vector<T>& expected) {
  using namespace ring_attention;
  DeviceTensor<T> src(expected.size());
  src.copy_from_host(expected);
  const T* src_ptr_before = src.data();

  DeviceTensor<T> dst(std::move(src));

  if (check(src.size() == 0, "moved-from size() == 0")) return 1;
  if (check(src.data() == nullptr, "moved-from data() == nullptr")) return 1;
  if (check(dst.data() == src_ptr_before, "moved-to inherits original pointer")) return 1;
  if (check(dst.size() == expected.size(), "moved-to size() preserved")) return 1;

  std::vector<T> got;
  dst.copy_to_host(got);
  for (std::size_t i = 0; i < expected.size(); ++i) {
    if (got[i] != expected[i]) {
      fprintf(stderr, "[%s] move-ctor mismatch at %zu\n", tag, i);
      return 1;
    }
  }
  return 0;
}

/// Move assignment. Exercises operator=(DeviceTensor&&) for type T.
/// Target has a pre-existing buffer that must be freed on assignment.
template <typename T>
int test_move_assignment(const char* tag, const std::vector<T>& expected) {
  using namespace ring_attention;
  DeviceTensor<T> src(expected.size());
  src.copy_from_host(expected);

  DeviceTensor<T> dst(8);  // pre-existing buffer — assignment must free it
  dst = std::move(src);

  if (check(src.size() == 0, "moved-from size() == 0 (assign)")) return 1;
  if (check(src.data() == nullptr, "moved-from data() == nullptr (assign)")) return 1;
  if (check(dst.size() == expected.size(), "moved-to size() updated")) return 1;

  std::vector<T> got;
  dst.copy_to_host(got);
  for (std::size_t i = 0; i < expected.size(); ++i) {
    if (got[i] != expected[i]) {
      fprintf(stderr, "[%s] move-assign mismatch at %zu\n", tag, i);
      return 1;
    }
  }
  return 0;
}

/// Move constructor for __half (separate because __half != comparison differs).
int test_move_constructor_half() {
  using namespace ring_attention;
  const float vals[] = {1.0f, -2.0f, 0.5f};
  constexpr int N = 3;
  std::vector<__half> expected(N);
  for (int i = 0; i < N; ++i) expected[i] = __float2half_rn(vals[i]);

  DeviceTensor<__half> src(N);
  src.copy_from_host(expected);
  const __half* src_ptr_before = src.data();

  DeviceTensor<__half> dst(std::move(src));

  if (check(src.size() == 0, "half moved-from size() == 0")) return 1;
  if (check(src.data() == nullptr, "half moved-from data() == nullptr")) return 1;
  if (check(dst.data() == src_ptr_before, "half moved-to inherits pointer")) return 1;

  std::vector<__half> got;
  dst.copy_to_host(got);
  for (int i = 0; i < N; ++i) {
    if (__half2float(got[i]) != __half2float(expected[i])) {
      fprintf(stderr, "half move-ctor mismatch at %d\n", i);
      return 1;
    }
  }
  return 0;
}

/// Move assignment for __half.
int test_move_assignment_half() {
  using namespace ring_attention;
  const float vals[] = {4.0f, 0.25f};
  constexpr int N = 2;
  std::vector<__half> expected(N);
  for (int i = 0; i < N; ++i) expected[i] = __float2half_rn(vals[i]);

  DeviceTensor<__half> src(N);
  src.copy_from_host(expected);

  DeviceTensor<__half> dst(8);
  dst = std::move(src);

  if (check(src.size() == 0, "half moved-from size() == 0 (assign)")) return 1;
  if (check(src.data() == nullptr, "half moved-from data() == nullptr (assign)")) return 1;
  if (check(dst.size() == static_cast<std::size_t>(N), "half moved-to size()")) return 1;

  std::vector<__half> got;
  dst.copy_to_host(got);
  for (int i = 0; i < N; ++i) {
    if (__half2float(got[i]) != __half2float(expected[i])) {
      fprintf(stderr, "half move-assign mismatch at %d\n", i);
      return 1;
    }
  }
  return 0;
}

/// The empty-tensor path skips cudaMalloc entirely (device_tensor.cu:14).
int test_empty_tensor() {
  using namespace ring_attention;

  DeviceTensor<float> a;
  if (check(a.size() == 0, "default-ctor size() == 0")) return 1;
  if (check(a.bytes() == 0, "default-ctor bytes() == 0")) return 1;
  if (check(a.data() == nullptr, "default-ctor data() == nullptr")) return 1;

  DeviceTensor<int> b(0);
  if (check(b.size() == 0, "explicit-(0)-ctor size() == 0")) return 1;
  if (check(b.data() == nullptr, "explicit-(0)-ctor data() == nullptr")) return 1;
  return 0;
}

/// Self-assignment guard: operator=(DeviceTensor&&) must be a no-op when
/// assigned to itself. Verifies the `if (this != &other)` branch is skipped.
int test_self_move_assignment() {
  using namespace ring_attention;
  const std::vector<int> expected = {5, 10, 15};
  DeviceTensor<int> d(expected.size());
  d.copy_from_host(expected);
  // Silence -Wself-move: cast through void* to avoid the compiler warning.
  d = std::move(*reinterpret_cast<DeviceTensor<int>*>(&d));
  // The self-move guard means d is unchanged.
  if (check(d.size() == expected.size(), "self-move size unchanged")) return 1;
  std::vector<int> got;
  d.copy_to_host(got);
  for (std::size_t i = 0; i < expected.size(); ++i) {
    if (got[i] != expected[i]) {
      fprintf(stderr, "self-move-assign mismatch at %zu\n", i);
      return 1;
    }
  }
  return 0;
}

}  // namespace

int main() {
  using namespace ring_attention;

  // --- Pointer-overload roundtrip for all four instantiations ---
  const std::vector<float> f_data = {1.0f, -2.5f, 3.14159f, 0.0f, 42.0f};
  const std::vector<int> i_data = {-7, 0, 1, 2, 1000000};
  const std::vector<unsigned> u_data = {0u, 1u, 2u, 0xDEADBEEFu, 0xFFFFFFFFu};

  if (test_roundtrip_ptr<float>("float", f_data)) return 1;
  if (test_roundtrip_ptr<int>("int", i_data)) return 1;
  if (test_roundtrip_ptr<unsigned>("uint", u_data)) return 1;
  if (test_roundtrip_half_ptr()) return 1;

  // --- Vector-overload roundtrip for all four instantiations ---
  if (test_roundtrip_vector_overload<float>("float-vec", f_data)) return 1;
  if (test_roundtrip_vector_overload<int>("int-vec", i_data)) return 1;
  if (test_roundtrip_vector_overload<unsigned>("uint-vec", u_data)) return 1;
  if (test_roundtrip_half_vector()) return 1;

  // --- Move constructor for all four instantiations ---
  if (test_move_constructor<float>("float", f_data)) return 1;
  if (test_move_constructor<int>("int", i_data)) return 1;
  if (test_move_constructor<unsigned>("uint", u_data)) return 1;
  if (test_move_constructor_half()) return 1;

  // --- Move assignment for all four instantiations ---
  if (test_move_assignment<float>("float", f_data)) return 1;
  if (test_move_assignment<int>("int", i_data)) return 1;
  if (test_move_assignment<unsigned>("uint", u_data)) return 1;
  if (test_move_assignment_half()) return 1;

  // --- Edge cases ---
  if (test_empty_tensor()) return 1;
  if (test_self_move_assignment()) return 1;

  printf("device_tensor OK\n");
  return 0;
}
