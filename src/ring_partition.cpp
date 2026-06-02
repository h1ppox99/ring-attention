/// @file
/// RingPartition implementation.
///
/// Two token-assignment modes:
///
///   Contiguous : rank r owns the contiguous block [r*chunk, (r+1)*chunk),
///                where chunk = seq / cp_size.
///
///   Zigzag (coarse) : the sequence is split into n_splits*cp_size chunks of
///                size seq/(n_splits*cp_size); rank r owns n_splits chunks.
///                Sub-groups are paired inward — (0, n-1), (1, n-2), … — so
///                each rank holds one early (cheap under causal) and one late
///                (expensive) chunk per pair, balancing per-rank workload.
///
///                Offset formula for sub-group sg, letting k = sg / 2:
///                  even sg: (k * cp_size + rank) * chunk
///                  odd  sg: ((n_splits-1-k) * cp_size + (cp_size-1-rank)) * chunk
///
/// Note: this is "coarse" zigzag (contiguous sub-groups). The Python
/// reference's `zigzag_indices` uses the finer scheme where each rank owns
/// n_splits positions in each macro-chunk of size n_splits*cp_size. Both
/// balance load equally well; coarse fits the single-`q_offset`-per-call
/// kernel API without scatter/gather.

#include "ring_partition.hpp"

#include <cassert>

namespace ring_attention {

RingPartition::RingPartition(int cp_size, int rank, int seq, Mode mode, int n_splits)
    : cp_size_(cp_size), rank_(rank), seq_(seq), n_splits_(n_splits), mode_(mode) {
  assert(cp_size > 0);
  assert(rank >= 0 && rank < cp_size);
  assert(seq % cp_size == 0);
  if (mode == Mode::Zigzag) {
    assert(n_splits >= 2);
    assert(seq % (n_splits * cp_size) == 0);
  }
}

int RingPartition::q_offset(int sg) const {
  if (mode_ == Mode::Contiguous) return rank_ * (seq_ / cp_size_);
  const int chunk = seq_ / (n_splits_ * cp_size_);
  const int k = sg / 2;
  if (sg % 2 == 0) return (k * cp_size_ + rank_) * chunk;
  return ((n_splits_ - 1 - k) * cp_size_ + (cp_size_ - 1 - rank_)) * chunk;
}

int RingPartition::k_offset_for_step(int step, int sg) const {
  // At step s this rank holds K/V that originated from rank (rank - s + cp_size) % cp_size.
  const int source = (rank_ - step % cp_size_ + cp_size_) % cp_size_;
  return k_offset_for_source(source, sg);
}

int RingPartition::k_offset_for_source(int source_rank, int sg) const {
  if (mode_ == Mode::Contiguous) return source_rank * (seq_ / cp_size_);
  // Zigzag: source's low chunk sits at slot `source_rank`; its high (mirror)
  // chunk at slot (2*cp_size - 1 - source_rank). Same mapping q_offset uses.
  const int chunk = seq_ / (n_splits_ * cp_size_);
  const int k = sg / 2;
  if (sg % 2 == 0) return (k * cp_size_ + source_rank) * chunk;
  return ((n_splits_ - 1 - k) * cp_size_ + (cp_size_ - 1 - source_rank)) * chunk;
}

int RingPartition::local_chunk_len(int /*sg*/) const {
  return (mode_ == Mode::Zigzag) ? seq_ / (n_splits_ * cp_size_) : seq_ / cp_size_;
}

int RingPartition::next_rank() const { return (rank_ + 1) % cp_size_; }
int RingPartition::prev_rank() const { return (rank_ - 1 + cp_size_) % cp_size_; }

int RingPartition::num_sub_groups() const { return (mode_ == Mode::Zigzag) ? n_splits_ : 1; }

}  // namespace ring_attention
