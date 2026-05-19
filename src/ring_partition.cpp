/// @file
/// RingPartition implementation. Contiguous mode only; Zigzag offsets stubbed.

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

int RingPartition::q_offset(int /*sg*/) const {
  // Contiguous: rank r owns tokens [r*chunk, (r+1)*chunk).
  // Zigzag sub-group offsets not yet implemented.
  return rank_ * (seq_ / cp_size_);
}

int RingPartition::k_offset_for_step(int step, int /*sg*/) const {
  // At step s this rank holds K/V that originated from rank (rank - s + cp_size) % cp_size.
  const int source = (rank_ - step % cp_size_ + cp_size_) % cp_size_;
  return source * (seq_ / cp_size_);
}

int RingPartition::local_chunk_len(int /*sg*/) const { return seq_ / cp_size_; }

int RingPartition::next_rank() const { return (rank_ + 1) % cp_size_; }
int RingPartition::prev_rank() const { return (rank_ - 1 + cp_size_) % cp_size_; }

int RingPartition::num_sub_groups() const { return (mode_ == Mode::Zigzag) ? 2 : 1; }

}  // namespace ring_attention
