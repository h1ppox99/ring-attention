#pragma once

/// @file
/// Ring2DSchedule: the hierarchical (2D) ring visitation order for context-
/// parallel attention on a clustered topology (N nodes × G GPUs/node).
/// Host-only (no CUDA, no MPI) — pure integer arithmetic, like RingPartition.
///
/// See docs/hierarchical_ring.md. The schedule keeps the heavy K/V rotation on
/// the fast intra-node ring and crosses the slow inter-node uplink only N-1
/// times per shard, cutting the round count from the flat ring's P-1 to P-G.
///
/// A global rank r maps to (node n, local g) with r = n*G + g (block layout:
/// ranks are contiguous per node). The orchestrator drives the kernel's K
/// offsets off `source(m, i)` — the global rank whose shard this GPU holds at
/// macro-step m, inner round i — via RingPartition::k_offset_for_source.

#include <vector>

namespace ring_attention {

class Ring2DSchedule {
 public:
  /// @param num_nodes      N: number of nodes (inter-node ring length).
  /// @param gpus_per_node  G: GPUs per node (intra-node ring length).
  /// @param node           n: this GPU's node index, in [0, N).
  /// @param local          g: this GPU's local index on its node, in [0, G).
  Ring2DSchedule(int num_nodes, int gpus_per_node, int node, int local);

  int num_nodes() const { return num_nodes_; }
  int gpus_per_node() const { return gpus_per_node_; }
  int cp_size() const { return num_nodes_ * gpus_per_node_; }  ///< P = N*G.
  int node() const { return node_; }
  int local() const { return local_; }
  int global_rank() const { return node_ * gpus_per_node_ + local_; }

  int num_macro_steps() const { return num_nodes_; }      ///< N outer (slow) steps.
  int num_inner_steps() const { return gpus_per_node_; }  ///< G inner (fast) rounds.

  /// Global rank whose shard this GPU holds at macro-step `m`, inner round `i`.
  /// The kernel's K offset for that shard is
  /// `RingPartition::k_offset_for_source(source(m, i), sg)`.
  int source(int m, int i) const { return trace_[m * gpus_per_node_ + i]; }

  /// Intra-node ring neighbors (global ranks). Shards rotate g -> (g+1)%G.
  int intra_next() const { return node_ * gpus_per_node_ + (local_ + 1) % gpus_per_node_; }
  int intra_prev() const {
    return node_ * gpus_per_node_ + (local_ - 1 + gpus_per_node_) % gpus_per_node_;
  }

  /// Inter-node ring neighbors (global ranks). The band rotates n -> (n+1)%N,
  /// keeping the local index fixed (GPU (n,g) pairs with ((n±1)%N, g)).
  int inter_next() const { return ((node_ + 1) % num_nodes_) * gpus_per_node_ + local_; }
  int inter_prev() const {
    return ((node_ - 1 + num_nodes_) % num_nodes_) * gpus_per_node_ + local_;
  }

  /// Inter-node rounds over the whole schedule, in shard-time units on the
  /// uplink: (N-1)*G == P - G. (Flat ring is P-1.)
  int inter_node_rounds() const { return (num_nodes_ - 1) * gpus_per_node_; }

 private:
  int num_nodes_, gpus_per_node_, node_, local_;
  std::vector<int> trace_;  ///< source held at each (m,i), flattened m*G + i.
};

}  // namespace ring_attention
