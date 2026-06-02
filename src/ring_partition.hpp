#pragma once

/// @file
/// RingPartition: maps ring step numbers to global token offsets and MPI neighbors.
/// Host-only (no CUDA); zero MPI dependency — pure integer arithmetic.

namespace ring_attention {

/// Describes how the global sequence is partitioned across a ring of cp_size ranks,
/// and computes the global token offsets needed for causal masking at each ring step.
class RingPartition {
 public:
  enum class Mode {
    Contiguous,  ///< Rank r owns tokens [r*chunk, (r+1)*chunk). Simple; unbalanced under causal.
    Zigzag,      ///< Rank r owns interleaved early+late tokens. Balanced under causal masking.
    Striped,     ///< Rank r owns tokens {r, r+cp_size, r+2*cp_size, ...} (token i -> rank i%cp).
                 ///< Like Zigzag it balances causal load, but with a *single* sub-group whose
                 ///< global positions form an affine map base + i*stride (stride = cp_size).
  };

  /// @param cp_size  Total ranks in the ring.
  /// @param rank     This rank's index, in [0, cp_size).
  /// @param seq      Global sequence length; must be divisible by cp_size (and by 2*cp_size
  ///                 for Zigzag mode).
  /// @param mode     Token assignment scheme.
  RingPartition(int cp_size, int rank, int seq, Mode mode = Mode::Contiguous);

  /// Global token position of the first query row in sub-group `sg`.
  /// Contiguous: sg is ignored (there is one sub-group). Zigzag: sg=0 = low, sg=1 = high.
  /// Striped: sg is ignored; the first local row sits at global position `rank`.
  ///
  /// The global position of local row `i` in sub-group `sg` is
  /// `q_offset(sg) + i * position_stride()`. For Contiguous/Zigzag the stride is 1
  /// (each sub-group is a contiguous run); for Striped it is `cp_size`.
  int q_offset(int sg = 0) const;

  /// Global token position of the first key row this rank holds at ring step `step`, sub-group
  /// `sg`. At step 0 each rank holds its own K/V; at step s it holds K/V from rank
  /// (rank - s + cp_size) % cp_size.
  int k_offset_for_step(int step, int sg = 0) const;

  /// Global token position of the first key row of the shard owned by `source_rank`,
  /// sub-group `sg`. The source-indexed primitive behind `k_offset_for_step`
  /// (which is `k_offset_for_source((rank - step + cp_size) % cp_size, sg)`). The
  /// hierarchical (2D) ring drives K offsets directly off the held shard's source
  /// rank, which is not a single-step function of this rank, so it calls this.
  int k_offset_for_source(int source_rank, int sg = 0) const;

  /// Number of local tokens per sub-group. Same for all sub-groups.
  int local_chunk_len(int sg = 0) const;

  int next_rank() const;  ///< Rank this rank sends K/V to: (rank + 1) % cp_size.
  int prev_rank() const;  ///< Rank this rank receives K/V from: (rank - 1 + cp_size) % cp_size.

  /// Number of sub-groups per rank: 1 for Contiguous, 2 for Zigzag, 1 for Striped.
  int num_sub_groups() const;

  /// Spacing (in global token positions) between consecutive local rows.
  /// Contiguous/Zigzag own contiguous runs -> 1. Striped interleaves tokens
  /// across ranks -> cp_size. Used by the kernel to recover a row's global
  /// position as `offset + local_index * position_stride()` for causal masking.
  int position_stride() const;

  int cp_size() const { return cp_size_; }
  int rank() const { return rank_; }
  int seq() const { return seq_; }
  Mode mode() const { return mode_; }

 private:
  int cp_size_, rank_, seq_;
  Mode mode_;
};

}  // namespace ring_attention
