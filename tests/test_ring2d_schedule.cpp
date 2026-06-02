/// @file
/// Unit tests for Ring2DSchedule — the hierarchical (2D) ring schedule.
/// CPU-only, no MPI, no GPU: pure integer arithmetic, like test_ring_partition.
///
/// The schedule is the regression-prone heart of the 2D ring: it decides which
/// global shard each GPU holds at every (macro, inner) step. These tests pin the
/// invariants the MPI orchestrator relies on — full coverage, the P-G inter-node
/// round count, and degeneracy to the flat ring — independent of any hardware.

#include <cassert>
#include <cstdio>
#include <set>
#include <vector>

#include "ring2d_schedule.hpp"

using ring_attention::Ring2DSchedule;

namespace {

void check(bool cond, const char* msg) {
  if (!cond) {
    fprintf(stderr, "FAIL: %s\n", msg);
    assert(false);
  }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Coverage: over all (macro, inner) steps, each GPU computes against every
/// global shard exactly once. This is the correctness property of the schedule
/// (online softmax is order-independent, so coverage ⇒ correct attention).
void test_full_coverage() {
  for (int N : {1, 2, 3, 4}) {
    for (int G : {1, 2, 3, 4}) {
      const int P = N * G;
      for (int n = 0; n < N; ++n) {
        for (int g = 0; g < G; ++g) {
          Ring2DSchedule s(N, G, n, g);
          check(s.num_macro_steps() == N, "num_macro_steps == N");
          check(s.num_inner_steps() == G, "num_inner_steps == G");
          std::set<int> seen;
          for (int m = 0; m < N; ++m)
            for (int i = 0; i < G; ++i) {
              const int src = s.source(m, i);
              check(src >= 0 && src < P, "source in [0,P)");
              check(seen.insert(src).second, "source visited at most once");
            }
          check(static_cast<int>(seen.size()) == P, "all P shards covered");
        }
      }
    }
  }
  printf("test_full_coverage OK\n");
}

/// At the very first step each GPU holds its own shard (== its global rank).
void test_first_step_is_own() {
  for (int N : {1, 2, 4}) {
    for (int G : {1, 2, 4}) {
      for (int n = 0; n < N; ++n)
        for (int g = 0; g < G; ++g) {
          Ring2DSchedule s(N, G, n, g);
          check(s.global_rank() == n * G + g, "global_rank == n*G+g");
          check(s.source(0, 0) == s.global_rank(), "source(0,0) == own rank");
        }
    }
  }
  printf("test_first_step_is_own OK\n");
}

/// Inter-node rounds (shard-time units on the slow uplink) = (N-1)*G = P - G.
/// This is the headline win over the flat ring's P-1 rounds.
void test_inter_node_round_count() {
  for (int N : {1, 2, 3, 4}) {
    for (int G : {1, 2, 3, 4}) {
      Ring2DSchedule s(N, G, 0, 0);
      const int P = N * G;
      check(s.inter_node_rounds() == (N - 1) * G, "inter rounds == (N-1)*G");
      check(s.inter_node_rounds() == P - G, "inter rounds == P - G");
    }
  }
  printf("test_inter_node_round_count OK\n");
}

/// N=1 (single node): no inter-node traffic, the schedule is a plain G-ring.
void test_single_node_degenerate() {
  const int N = 1, G = 4;
  for (int g = 0; g < G; ++g) {
    Ring2DSchedule s(N, G, 0, g);
    check(s.inter_node_rounds() == 0, "N=1: zero inter-node rounds");
    check(s.num_macro_steps() == 1, "N=1: one macro step");
    check(s.num_inner_steps() == G, "N=1: G inner steps");
    std::set<int> seen;
    for (int i = 0; i < G; ++i) seen.insert(s.source(0, i));
    check(static_cast<int>(seen.size()) == G, "N=1: inner loop covers all G shards");
  }
  printf("test_single_node_degenerate OK\n");
}

/// G=1 (one GPU/node): no intra-node tier, the schedule must reproduce the flat
/// ring exactly — at macro m the held source is (rank - m + N) % N, matching
/// RingPartition::k_offset_for_step's source. Inter rounds = N-1 = P-1.
void test_flat_ring_degenerate() {
  const int G = 1;
  for (int N : {2, 3, 5}) {
    for (int n = 0; n < N; ++n) {
      Ring2DSchedule s(N, G, n, 0);
      check(s.inter_node_rounds() == N - 1, "G=1: inter rounds == N-1 == P-1");
      check(s.num_inner_steps() == 1, "G=1: single inner step");
      for (int m = 0; m < N; ++m) {
        const int expected = (n - m + N) % N;  // flat-ring source at step m
        char msg[64];
        std::snprintf(msg, sizeof(msg), "G=1 flat ring n=%d m=%d", n, m);
        check(s.source(m, 0) == expected, msg);
      }
    }
  }
  printf("test_flat_ring_degenerate OK\n");
}

/// Neighbor rings are consistent: my intra_next's intra_prev is me, and likewise
/// for inter neighbors. Also checks neighbors are valid distinct global ranks.
void test_neighbor_ring_consistency() {
  const int N = 3, G = 4;
  for (int n = 0; n < N; ++n)
    for (int g = 0; g < G; ++g) {
      Ring2DSchedule s(N, G, n, g);
      // Intra neighbors stay on the same node, rotate the local index.
      const int in = s.intra_next(), ip = s.intra_prev();
      check(in / G == n && ip / G == n, "intra neighbors on same node");
      Ring2DSchedule sn(N, G, in / G, in % G);
      check(sn.intra_prev() == s.global_rank(), "intra_next.intra_prev == self");
      // Inter neighbors keep the local index, rotate the node.
      const int en = s.inter_next(), ep = s.inter_prev();
      check(en % G == g && ep % G == g, "inter neighbors keep local index");
      Ring2DSchedule se(N, G, en / G, en % G);
      check(se.inter_prev() == s.global_rank(), "inter_next.inter_prev == self");
    }
  printf("test_neighbor_ring_consistency OK\n");
}

/// Exact trace lock for N=2, G=2 (P=4): pins the precise visitation order so a
/// future refactor of the rotation rule can't silently change semantics.
void test_exact_trace_n2_g2() {
  const int N = 2, G = 2;
  // sources[rank][m*G + i]
  const int expected[4][4] = {
      {0, 1, 2, 3},  // rank 0 = (node 0, local 0)
      {1, 0, 3, 2},  // rank 1 = (node 0, local 1)
      {2, 3, 0, 1},  // rank 2 = (node 1, local 0)
      {3, 2, 1, 0},  // rank 3 = (node 1, local 1)
  };
  for (int r = 0; r < 4; ++r) {
    Ring2DSchedule s(N, G, r / G, r % G);
    for (int m = 0; m < N; ++m)
      for (int i = 0; i < G; ++i) {
        char msg[48];
        std::snprintf(msg, sizeof(msg), "trace rank=%d m=%d i=%d", r, m, i);
        check(s.source(m, i) == expected[r][m * G + i], msg);
      }
  }
  printf("test_exact_trace_n2_g2 OK\n");
}

}  // namespace

int main() {
  test_full_coverage();
  test_first_step_is_own();
  test_inter_node_round_count();
  test_single_node_degenerate();
  test_flat_ring_degenerate();
  test_neighbor_ring_consistency();
  test_exact_trace_n2_g2();
  printf("All ring2d_schedule tests passed.\n");
  return 0;
}
