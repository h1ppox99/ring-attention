/// @file
/// Unit tests for RingPartition. CPU-only, no MPI, no GPU.

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <set>
#include <vector>

#include "ring_partition.hpp"

using ring_attention::RingPartition;

namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

void check(bool cond, const char* msg) {
  if (!cond) {
    fprintf(stderr, "FAIL: %s\n", msg);
    assert(false);
  }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Basic geometry: cp_size=1 is the identity case.
void test_single_rank() {
  RingPartition p(1, 0, 256);
  check(p.q_offset() == 0, "single rank q_offset");
  check(p.local_chunk_len() == 256, "single rank chunk_len");
  check(p.k_offset_for_step(0) == 0, "single rank step 0");
  check(p.next_rank() == 0, "single rank next");
  check(p.prev_rank() == 0, "single rank prev");
  check(p.num_sub_groups() == 1, "single rank sub_groups");
  printf("test_single_rank OK\n");
}

/// cp_size=4, seq=8 (chunk=2): exhaustive check of q_offset and k_offset_for_step.
void test_4ranks_seq8() {
  const int cp_size = 4, seq = 8, chunk = 2;
  for (int r = 0; r < cp_size; ++r) {
    RingPartition p(cp_size, r, seq);
    check(p.q_offset() == r * chunk, "q_offset");
    check(p.local_chunk_len() == chunk, "chunk_len");
    check(p.next_rank() == (r + 1) % cp_size, "next_rank");
    check(p.prev_rank() == (r - 1 + cp_size) % cp_size, "prev_rank");
    check(p.num_sub_groups() == 1, "contiguous sub_groups");

    for (int step = 0; step < cp_size; ++step) {
      const int expected_source = (r - step + cp_size) % cp_size;
      const int expected_off = expected_source * chunk;
      char msg[64];
      std::snprintf(msg, sizeof(msg), "rank %d step %d k_offset", r, step);
      check(p.k_offset_for_step(step) == expected_off, msg);
    }
  }
  printf("test_4ranks_seq8 OK\n");
}

/// After cp_size steps, every rank's K/V has been seen exactly once.
/// Verified for each (cp_size, rank) combination.
void test_full_ring_coverage() {
  for (int cp_size : {2, 3, 4}) {
    const int seq = cp_size * 16;  // chunk = 16
    const int chunk = seq / cp_size;
    for (int r = 0; r < cp_size; ++r) {
      RingPartition p(cp_size, r, seq);
      std::set<int> seen;
      for (int step = 0; step < cp_size; ++step) seen.insert(p.k_offset_for_step(step));
      // Must have visited all cp_size distinct chunk starts.
      check(static_cast<int>(seen.size()) == cp_size, "full coverage: distinct count");
      for (int src = 0; src < cp_size; ++src)
        check(seen.count(src * chunk) == 1, "full coverage: each chunk present");
    }
  }
  printf("test_full_ring_coverage OK\n");
}

/// Step 0 always returns this rank's own K/V (k_offset == q_offset).
void test_step0_is_own_kv() {
  for (int cp_size : {1, 2, 4}) {
    const int seq = cp_size * 32;
    for (int r = 0; r < cp_size; ++r) {
      RingPartition p(cp_size, r, seq);
      check(p.k_offset_for_step(0) == p.q_offset(), "step 0 == q_offset");
    }
  }
  printf("test_step0_is_own_kv OK\n");
}

/// Contiguous mode: num_sub_groups is 1. Zigzag mode: num_sub_groups is 2.
void test_num_sub_groups() {
  RingPartition c(4, 0, 32, RingPartition::Mode::Contiguous);
  check(c.num_sub_groups() == 1, "contiguous -> 1 sub_group");

  RingPartition z(4, 0, 32, RingPartition::Mode::Zigzag);
  check(z.num_sub_groups() == 2, "zigzag -> 2 sub_groups");
  printf("test_num_sub_groups OK\n");
}

/// Ring topology: next/prev form a consistent ring.
void test_ring_topology() {
  const int cp_size = 5;
  for (int r = 0; r < cp_size; ++r) {
    RingPartition p(cp_size, r, cp_size * 8);
    const int nxt = p.next_rank();
    const int prv = p.prev_rank();
    RingPartition q(cp_size, nxt, cp_size * 8);
    // next's prev must be this rank.
    check(q.prev_rank() == r, "ring topology: next.prev == self");
    (void)prv;
  }
  printf("test_ring_topology OK\n");
}

/// local_chunk_len == seq / cp_size for all ranks and sizes.
void test_local_chunk_len() {
  for (int cp_size : {1, 2, 4}) {
    const int seq = cp_size * 32;
    for (int r = 0; r < cp_size; ++r) {
      RingPartition p(cp_size, r, seq);
      check(p.local_chunk_len() == seq / cp_size, "local_chunk_len == seq/cp_size");
    }
  }
  printf("test_local_chunk_len OK\n");
}

/// Getters return the values passed to the constructor.
void test_getters() {
  RingPartition c(4, 2, 128, RingPartition::Mode::Contiguous);
  check(c.cp_size() == 4, "getter cp_size");
  check(c.rank() == 2, "getter rank");
  check(c.seq() == 128, "getter seq");
  check(c.mode() == RingPartition::Mode::Contiguous, "getter mode contiguous");

  RingPartition z(4, 0, 128, RingPartition::Mode::Zigzag);
  check(z.mode() == RingPartition::Mode::Zigzag, "getter mode zigzag");
  printf("test_getters OK\n");
}

/// Zigzag mode: each rank's two sub-groups live in distinct contiguous chunks
/// of size seq/(2*cp_size). Exhaustive check for cp_size=4, seq=16 (chunk=2).
void test_zigzag_offsets_4ranks() {
  const int cp_size = 4, seq = 16, chunk = 2;  // 2*cp_size = 8 chunks of size 2
  for (int r = 0; r < cp_size; ++r) {
    RingPartition p(cp_size, r, seq, RingPartition::Mode::Zigzag);
    check(p.num_sub_groups() == 2, "zigzag: 2 sub-groups");
    check(p.local_chunk_len() == chunk, "zigzag: chunk = seq/(2*cp_size)");
    // sub-group 0 ("low") is at rank's natural chunk slot.
    check(p.q_offset(0) == r * chunk, "zigzag q_offset(low)");
    // sub-group 1 ("high") is at the mirror slot in the late half.
    check(p.q_offset(1) == (2 * cp_size - 1 - r) * chunk, "zigzag q_offset(high)");
    for (int step = 0; step < cp_size; ++step) {
      const int src = (r - step + cp_size) % cp_size;
      char m0[80], m1[80];
      std::snprintf(m0, sizeof(m0), "zigzag r=%d step=%d k_off(low)", r, step);
      std::snprintf(m1, sizeof(m1), "zigzag r=%d step=%d k_off(high)", r, step);
      check(p.k_offset_for_step(step, 0) == src * chunk, m0);
      check(p.k_offset_for_step(step, 1) == (2 * cp_size - 1 - src) * chunk, m1);
    }
  }
  printf("test_zigzag_offsets_4ranks OK\n");
}

/// Zigzag: at step 0 each sub-group's K must match the rank's own Q sub-group.
void test_zigzag_step0_own() {
  for (int cp_size : {1, 2, 4}) {
    const int seq = cp_size * 32;
    for (int r = 0; r < cp_size; ++r) {
      RingPartition p(cp_size, r, seq, RingPartition::Mode::Zigzag);
      check(p.k_offset_for_step(0, 0) == p.q_offset(0), "zigzag step 0 sg=0");
      check(p.k_offset_for_step(0, 1) == p.q_offset(1), "zigzag step 0 sg=1");
    }
  }
  printf("test_zigzag_step0_own OK\n");
}

/// Zigzag: the union of sub-groups across all ranks covers every chunk exactly
/// once — no overlaps, no gaps. This is the partition correctness property.
void test_zigzag_partition_cover() {
  const int cp_size = 4, seq = 32, chunk = 4;  // 8 chunks of size 4
  std::set<int> seen;
  for (int r = 0; r < cp_size; ++r) {
    RingPartition p(cp_size, r, seq, RingPartition::Mode::Zigzag);
    for (int sg = 0; sg < 2; ++sg) {
      const int off = p.q_offset(sg);
      check(seen.count(off) == 0, "zigzag: no duplicate chunk in partition");
      seen.insert(off);
    }
  }
  check(static_cast<int>(seen.size()) == 2 * cp_size, "zigzag: all chunks covered");
  for (int c = 0; c < 2 * cp_size; ++c)
    check(seen.count(c * chunk) == 1, "zigzag: every chunk position present");
  printf("test_zigzag_partition_cover OK\n");
}

}  // namespace

int main() {
  test_single_rank();
  test_4ranks_seq8();
  test_full_ring_coverage();
  test_step0_is_own_kv();
  test_num_sub_groups();
  test_ring_topology();
  test_local_chunk_len();
  test_getters();
  test_zigzag_offsets_4ranks();
  test_zigzag_step0_own();
  test_zigzag_partition_cover();
  printf("All ring_partition tests passed.\n");
  return 0;
}
