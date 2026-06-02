/// @file
/// RingPartition implementation.
///
/// Two token-assignment modes:
///
///   Contiguous : rank r owns the contiguous block [r*chunk, (r+1)*chunk),
///                where chunk = seq / cp_size.
///
///   Zigzag (coarse) : the sequence is split into 2*cp_size chunks of size
///                seq/(2*cp_size); rank r owns chunks r and (2*cp_size-1-r).
///                The two chunks live in distinct *sub-groups* — both are
///                contiguous in global coords, but at very different
///                positions (one in the early half, one in the late half).
///                Under causal masking this evens out the per-rank workload.
///
/// Note: this is "coarse" zigzag (2 contiguous sub-groups). The Python
/// reference's `zigzag_indices` uses the finer scheme where each rank owns
/// 2 positions in each macro-chunk of size 2*cp_size. Both balance load
/// equally well; coarse is what fits the single-`q_offset`-per-call kernel
/// API without scatter/gather.

#include "ring_partition.hpp"

#include <cassert>

namespace ring_attention {

RingPartition::RingPartition(int cp_size, int rank, int seq, Mode mode)
    : cp_size_(cp_size), rank_(rank), seq_(seq), mode_(mode) {
  assert(cp_size > 0);
  assert(rank >= 0 && rank < cp_size);
  assert(seq % cp_size == 0);
  if (mode == Mode::Zigzag) assert(seq % (2 * cp_size) == 0);
}

int RingPartition::q_offset(int sg) const {
  if (mode_ == Mode::Contiguous) return rank_ * (seq_ / cp_size_);
  // Zigzag: sub-group 0 is the "early" chunk; sub-group 1 is its mirror.
  const int chunk = seq_ / (2 * cp_size_);
  return (sg == 0 ? rank_ : (2 * cp_size_ - 1 - rank_)) * chunk;
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
  const int chunk = seq_ / (2 * cp_size_);
  return (sg == 0 ? source_rank : (2 * cp_size_ - 1 - source_rank)) * chunk;
}

int RingPartition::local_chunk_len(int /*sg*/) const {
  // Sub-groups always have equal length, both in contiguous (one group of
  // seq/cp_size) and zigzag (two groups of seq/(2*cp_size) each).
  return (mode_ == Mode::Zigzag) ? seq_ / (2 * cp_size_) : seq_ / cp_size_;
}

int RingPartition::next_rank() const { return (rank_ + 1) % cp_size_; }
int RingPartition::prev_rank() const { return (rank_ - 1 + cp_size_) % cp_size_; }

int RingPartition::num_sub_groups() const { return (mode_ == Mode::Zigzag) ? 2 : 1; }

}  // namespace ring_attention
