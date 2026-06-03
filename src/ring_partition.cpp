/// @file
/// RingPartition implementation.
///
/// Three token-assignment modes:
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
///   Striped : rank r owns tokens {r, r+cp_size, r+2*cp_size, ...}, i.e. token
///                i is assigned to rank (i % cp_size). A single sub-group, but
///                its global positions are *not* contiguous: local row i sits at
///                global position r + i*cp_size. This is an affine map with
///                base = r and stride = cp_size, so it still fits the
///                single-offset kernel API — the kernel just needs the stride
///                (see `position_stride`). Like Zigzag it balances causal load,
///                and additionally makes the per-ring-step mask near-uniform.
///
/// Note: Zigzag here is "coarse" (contiguous sub-groups). The Python
/// reference's `zigzag_indices` uses the finer scheme where each rank owns
/// n_splits positions in each macro-chunk of size n_splits*cp_size. Both
/// balance load equally well; coarse fits the single-`q_offset`-per-call
/// kernel API without scatter/gather. Striped achieves the fine-grained
/// interleave via the stride instead of via scatter/gather.

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
  // Striped: one sub-group; local row i is at global position rank_ + i*cp_size
  // (see position_stride). The base offset is therefore just the rank.
  if (mode_ == Mode::Striped) return rank_;
  // Zigzag: sub-group sg is paired inward (k = sg/2); even sg = early chunk,
  // odd sg = its mirror in the late half.
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
  // Zigzag: source's chunks are paired inward, same mapping q_offset uses.
  // Striped: the held chunk's local row j is at global position source + j*cp_size.
  if (mode_ == Mode::Striped) return source_rank;
  const int chunk = seq_ / (n_splits_ * cp_size_);
  const int k = sg / 2;
  if (sg % 2 == 0) return (k * cp_size_ + source_rank) * chunk;
  return ((n_splits_ - 1 - k) * cp_size_ + (cp_size_ - 1 - source_rank)) * chunk;
}

int RingPartition::local_chunk_len(int /*sg*/) const {
  // Contiguous and striped own one group of seq/cp_size; zigzag owns n_splits
  // groups of seq/(n_splits*cp_size) each.
  return (mode_ == Mode::Zigzag) ? seq_ / (n_splits_ * cp_size_) : seq_ / cp_size_;
}

int RingPartition::next_rank() const { return (rank_ + 1) % cp_size_; }
int RingPartition::prev_rank() const { return (rank_ - 1 + cp_size_) % cp_size_; }

int RingPartition::num_sub_groups() const { return (mode_ == Mode::Zigzag) ? n_splits_ : 1; }

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
