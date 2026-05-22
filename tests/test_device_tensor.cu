/// @file
/// Unit tests for DeviceTensor: ctor/dtor, both copy_from_host / copy_to_host
/// overloads, move semantics, and all three template instantiations
/// (float, int, unsigned int).
///
/// Exit code convention matches the other CUDA tests in this directory: print
/// "mismatch ..." to stderr and return 1 on failure, "device_tensor OK" + 0
/// on success.

#include <cstdint>
#include <cstdio>
#include <vector>

#include "device_tensor.hpp"

namespace {

// Tiny assertion helper to keep the body terse. Returns 1 on failure so the
// caller can do `if (check(...)) return 1;`. We deliberately do not use any
// gtest-style framework — tests/CMakeLists.txt links plain executables.
int check(bool cond, const char* what) {
  if (!cond) {
    fprintf(stderr, "FAIL: %s\n", what);
    return 1;
  }
  return 0;
}

/// Round-trip a host vector through a DeviceTensor<T> using the pointer
/// overloads, verify the data survives the trip unchanged.
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
      fprintf(stderr, "[%s] mismatch at %zu: got %lld, expected %lld\n", tag, i,
              static_cast<long long>(got[i]), static_cast<long long>(expected[i]));
      return 1;
    }
  }
  return 0;
}

/// Exercises the std::vector overloads of copy_from_host / copy_to_host
/// (device_tensor.cu:56-65). The key behavior to verify is that copy_to_host
/// *resizes* the destination vector — so we start with an empty one.
int test_roundtrip_vector_overload() {
  using namespace ring_attention;
  const std::vector<int> expected = {-3, 0, 7, 42, -1};

  DeviceTensor<int> d(expected.size());
  d.copy_from_host(expected);  // vector overload — assert checks size match

  std::vector<int> got;  // intentionally empty; copy_to_host must resize it
  d.copy_to_host(got);
  if (check(got.size() == expected.size(), "vector overload resized destination")) return 1;
  for (std::size_t i = 0; i < expected.size(); ++i) {
    if (got[i] != expected[i]) {
      fprintf(stderr, "vector-overload mismatch at %zu: got %d, expected %d\n", i, got[i],
              expected[i]);
      return 1;
    }
  }
  return 0;
}

/// Exercises DeviceTensor(DeviceTensor&&). The key invariants:
///   (1) The moved-from object is in a valid empty state (size==0, data==null)
///       so its destructor is a no-op rather than a double-free of GPU memory.
///   (2) The moved-to object owns the original GPU buffer and round-trips
///       data correctly — i.e. the move actually transferred ownership rather
///       than constructing a fresh empty tensor.
int test_move_constructor() {
  using namespace ring_attention;
  const std::vector<float> expected = {1.f, 2.f, 3.f, 4.f};

  DeviceTensor<float> src(expected.size());
  src.copy_from_host(expected);
  const float* src_ptr_before = src.data();

  DeviceTensor<float> dst(std::move(src));  // ← the move ctor under test

  // (1) moved-from invariants — destructor of `src` must be a no-op.
  if (check(src.size() == 0, "moved-from size() == 0")) return 1;
  if (check(src.data() == nullptr, "moved-from data() == nullptr")) return 1;

  // (2) moved-to owns the original buffer and still has the data.
  if (check(dst.data() == src_ptr_before, "moved-to inherits original GPU pointer")) return 1;
  if (check(dst.size() == expected.size(), "moved-to size() preserved")) return 1;
  std::vector<float> got;
  dst.copy_to_host(got);
  for (std::size_t i = 0; i < expected.size(); ++i) {
    if (got[i] != expected[i]) {
      fprintf(stderr, "move-ctor mismatch at %zu\n", i);
      return 1;
    }
  }
  return 0;
}

/// Exercises operator=(DeviceTensor&&). Same two invariants as the move ctor,
/// plus a third: the previous contents of the assignment target must be
/// cudaFree'd before being overwritten — otherwise we leak GPU memory on
/// every assignment. We can't directly observe a leak from a test, but we can
/// at least confirm the assignment succeeds when the target already owns a
/// non-empty buffer (which is the path that exercises the cudaFree branch at
/// device_tensor.cu:36).
int test_move_assignment() {
  using namespace ring_attention;
  const std::vector<int> expected = {10, 20, 30};

  DeviceTensor<int> src(expected.size());
  src.copy_from_host(expected);

  DeviceTensor<int> dst(8);  // pre-existing buffer — assignment must free it
  dst = std::move(src);      // ← the move assignment under test

  // moved-from invariants
  if (check(src.size() == 0, "moved-from size() == 0 (assign)")) return 1;
  if (check(src.data() == nullptr, "moved-from data() == nullptr (assign)")) return 1;

  // moved-to has the new data, not the old size-8 buffer.
  if (check(dst.size() == expected.size(), "moved-to size() updated (assign)")) return 1;
  std::vector<int> got;
  dst.copy_to_host(got);
  for (std::size_t i = 0; i < expected.size(); ++i) {
    if (got[i] != expected[i]) {
      fprintf(stderr, "move-assign mismatch at %zu\n", i);
      return 1;
    }
  }
  return 0;
}

/// The empty-tensor path skips cudaMalloc entirely (device_tensor.cu:14).
/// Both a default-constructed tensor and an explicit-(0)-constructed one must
/// satisfy: size()==0, data()==nullptr, and destruction is safe with no
/// CUDA calls. The latter case proves the `if (count_ > 0)` guard works.
int test_empty_tensor() {
  using namespace ring_attention;

  DeviceTensor<float> a;  // default ctor — never even enters the count_>0 branch
  if (check(a.size() == 0, "default-ctor size() == 0")) return 1;
  if (check(a.bytes() == 0, "default-ctor bytes() == 0")) return 1;
  if (check(a.data() == nullptr, "default-ctor data() == nullptr")) return 1;

  DeviceTensor<int> b(0);  // explicit zero — the guarded-skip path
  if (check(b.size() == 0, "explicit-(0)-ctor size() == 0")) return 1;
  if (check(b.data() == nullptr, "explicit-(0)-ctor data() == nullptr")) return 1;

  // Destructors run at scope exit; if either tried to cudaFree(nullptr) on a
  // bogus pointer we'd see a CUDA error on the next API call. Force-flush:
  return 0;
}

}  // namespace

int main() {
  using namespace ring_attention;

  // Smoke matrix: round-trip on all three template instantiations to ensure
  // every `template class DeviceTensor<T>` in device_tensor.cu is exercised.
  const std::vector<float> f_data = {1.0f, -2.5f, 3.14159f, 0.0f, 42.0f};
  const std::vector<int> i_data = {-7, 0, 1, 2, 1000000};
  const std::vector<unsigned> u_data = {0u, 1u, 2u, 0xDEADBEEFu, 0xFFFFFFFFu};

  if (test_roundtrip_ptr<float>("float", f_data)) return 1;
  if (test_roundtrip_ptr<int>("int", i_data)) return 1;
  if (test_roundtrip_ptr<unsigned>("uint", u_data)) return 1;
  if (test_roundtrip_vector_overload()) return 1;
  if (test_move_constructor()) return 1;
  if (test_move_assignment()) return 1;
  if (test_empty_tensor()) return 1;

  printf("device_tensor OK\n");
  return 0;
}
