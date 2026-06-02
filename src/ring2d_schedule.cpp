/// @file
/// Ring2DSchedule implementation.
///
/// The visitation order has a closed form, chosen so the inter-node transfer can
/// overlap the entire inner loop (see docs/hierarchical_ring.md):
///
///   - Each GPU has a per-macro "seed" shard it holds at inner round 0. The
///     inter-node pass ships that seed (known at macro start, so the transfer
///     runs concurrently with the inner loop) to the paired GPU on the next
///     node: seed(m+1, n, g) = seed(m, (n-1+N)%N, g), seed(0, n, g) = n*G+g.
///     Hence seed(m, n, g) = ((n-m+N)%N)*G + g.
///   - Within a macro-step the band rotates around the intra-node ring: at inner
///     round i, GPU (n,g) holds the seed of its i-th intra-node predecessor,
///     (n, (g-i+G)%G).
///
/// Combining: source(m,i) = ((n-m+N)%N)*G + (g-i+G)%G. Over all (m,i) the node
/// term ranges over all N nodes and the local term over all G GPUs independently,
/// so every one of the P shards is visited exactly once (coverage). The MPI
/// orchestrator mirrors these two rotation rules, so the integer schedule and
/// the actual data movement stay in lock-step.

#include "ring2d_schedule.hpp"

#include <cassert>

namespace ring_attention {

Ring2DSchedule::Ring2DSchedule(int num_nodes, int gpus_per_node, int node, int local)
    : num_nodes_(num_nodes), gpus_per_node_(gpus_per_node), node_(node), local_(local) {
  assert(num_nodes > 0 && gpus_per_node > 0);
  assert(node >= 0 && node < num_nodes);
  assert(local >= 0 && local < gpus_per_node);

  const int N = num_nodes_, G = gpus_per_node_;
  trace_.resize(static_cast<std::size_t>(N) * G);
  for (int m = 0; m < N; ++m)
    for (int i = 0; i < G; ++i) {
      const int node_term = ((node_ - m + N) % N) * G;
      const int local_term = (local_ - i + G) % G;
      trace_[m * G + i] = node_term + local_term;
    }
}

}  // namespace ring_attention
