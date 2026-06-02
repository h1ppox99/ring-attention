/// @file
/// RingPartition implementation.
///
/// Three token-assignment modes:
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
///   Striped : rank r owns tokens {r, r+cp_size, r+2*cp_size, ...}, i.e. token
///                i is assigned to rank (i % cp_size). A single sub-group, but
///                its global positions are *not* contiguous: local row i sits at
///                global position r + i*cp_size. This is an affine map with
///                base = r and stride = cp_size, so it still fits the
///                single-offset kernel API — the kernel just needs the stride
///                (see `position_stride`). Like Zigzag it balances causal load,
///                and additionally makes the per-ring-step mask near-uniform.
///
/// Note: Zigzag here is "coarse" (2 contiguous sub-groups). The Python
/// reference's `zigzag_indices` uses the finer scheme where each rank owns
/// 2 positions in each macro-chunk of size 2*cp_size. Both balance load
/// equally well; coarse is what fits the single-`q_offset`-per-call kernel
/// API without scatter/gather. Striped achieves the fine-grained interleave
/// via the stride instead of via scatter/gather.

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
  // Striped: one sub-group; local row i is at global position rank_ + i*cp_size
  // (see position_stride). The base offset is therefore just the rank.
  if (mode_ == Mode::Striped) return rank_;
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
  // Striped: the held chunk's local row j is at global position source + j*cp_size.
  if (mode_ == Mode::Striped) return source_rank;
  const int chunk = seq_ / (2 * cp_size_);
  return (sg == 0 ? source_rank : (2 * cp_size_ - 1 - source_rank)) * chunk;
}

int RingPartition::local_chunk_len(int /*sg*/) const {
  // Sub-groups always have equal length: contiguous and striped own one group
  // of seq/cp_size; zigzag owns two groups of seq/(2*cp_size) each.
  return (mode_ == Mode::Zigzag) ? seq_ / (2 * cp_size_) : seq_ / cp_size_;
}

int RingPartition::next_rank() const { return (rank_ + 1) % cp_size_; }
int RingPartition::prev_rank() const { return (rank_ - 1 + cp_size_) % cp_size_; }

int RingPartition::num_sub_groups() const { return (mode_ == Mode::Zigzag) ? 2 : 1; }

int RingPartition::position_stride() const {
  // Global position of local row i in a sub-group is: offset + i * stride.
  // Contiguous and Zigzag each own a *contiguous* run of global positions, so
  // consecutive local rows are consecutive global positions -> stride 1.
  switch (mode_) {
    case Mode::Contiguous:
    case Mode::Zigzag:
      return 1;
    case Mode::Striped:
      // Striped assigns token i to rank (i % cp_size), so rank r owns global
      // positions r, r+cp_size, r+2*cp_size, … — consecutive local rows are
      // exactly cp_size apart in global-position space.
      return cp_size_;
  }
  return 1;
}

}  // namespace ring_attention
