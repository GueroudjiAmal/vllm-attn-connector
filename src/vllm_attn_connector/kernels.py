"""Exact decode-step attention over the paged prompt keys.

At decode step ``t`` a request contributes exactly one query token, so the
attention it pays to the prompt is

    a_t = softmax( q_t . K_prompt^T / sqrt(d) )        # [num_heads, prompt_len]

This is *recomputed*, not extracted. FlashAttention tiles the softmax and
discards the scores, so there is nothing to read back -- but K persists in the
cache, and one decode row is a matvec, ``O(T)``, not the ``O(T^2)`` of prefill.

The kernel reads the paged cache directly rather than gathering K into a dense
buffer first: a gather would add a full write-and-reread of the keys, roughly
doubling what is already the dominant memory traffic of decode.

One program handles one (head, physical block), walking the block in
fixed-size tiles. The tile is constant rather than the block size itself:
``tl.arange`` needs a power-of-two extent, vLLM does not guarantee one of the
block size, and a hybrid model's attention page can be very large (1056 tokens
on Qwen3.5, sized to match its gated-delta-net state). Loads stay contiguous
within a tile, and the tail is masked.
"""

from __future__ import annotations

import math

import torch

try:
    from vllm.triton_utils import tl, triton

    HAS_TRITON = True
except Exception:  # pragma: no cover - exercised only outside vLLM
    try:
        import triton
        import triton.language as tl

        HAS_TRITON = True
    except Exception:
        triton = None  # type: ignore[assignment]
        tl = None  # type: ignore[assignment]
        HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _qk_kernel(
        q_ptr, k_ptr, block_table_ptr, out_ptr,
        n_keys, scale,
        q_sh, q_sd,
        k_sb, k_sn, k_sh, k_sd,
        out_sh,
        BLOCK_SIZE: tl.constexpr,
        GROUP: tl.constexpr,          # query heads per KV head (GQA fan-out)
        HEAD_SIZE: tl.constexpr,
        PAD_D: tl.constexpr,
        TILE: tl.constexpr,           # tokens per inner tile; a power of 2
        N_TILES: tl.constexpr,        # ceil(BLOCK_SIZE / TILE)
    ):
        h = tl.program_id(0)
        blk = tl.program_id(1)
        kvh = h // GROUP

        d = tl.arange(0, PAD_D)
        d_ok = d < HEAD_SIZE
        q = tl.load(q_ptr + h * q_sh + d * q_sd, mask=d_ok, other=0.0).to(tl.float32)

        # The block is walked in FIXED-SIZE TILES rather than as one tile.
        #
        # `tl.arange` needs a power-of-two extent, and the paged block size is
        # not one in general -- vLLM sizes it per model and backend. Rounding
        # the block up to a power of two works while blocks are small, but a
        # hybrid model (full attention interleaved with gated-delta-net layers)
        # gets its attention page sized to match the GDN state page: Qwen3.5
        # reports block_size=1056, which rounds to 2048 and asks for a
        # [2048, HEAD_SIZE] float tile -- megabytes per program. The kernel
        # then does not write its output and the scores come back NaN.
        #
        # A constant tile keeps register pressure fixed whatever the block
        # size. TILE divides evenly into the arange, and every token is still
        # visited exactly once because the loop covers [0, BLOCK_SIZE).
        block_id = tl.load(block_table_ptr + blk).to(tl.int64)

        for t0 in tl.static_range(0, N_TILES):
            n = t0 * TILE + tl.arange(0, TILE)
            tok = blk * BLOCK_SIZE + n
            tok_ok = (n < BLOCK_SIZE) & (tok < n_keys)

            k = tl.load(
                k_ptr + block_id * k_sb + n[:, None] * k_sn + kvh * k_sh
                + d[None, :] * k_sd,
                mask=tok_ok[:, None] & d_ok[None, :],
                other=0.0,
            ).to(tl.float32)

            logits = tl.sum(k * q[None, :], axis=1) * scale
            tl.store(out_ptr + h * out_sh + tok, logits, mask=tok_ok)


def decode_attention(
    q: torch.Tensor,
    k_view: torch.Tensor,
    block_table: torch.Tensor,
    n_keys: int,
    *,
    num_query_heads: int,
    scratch: torch.Tensor | None = None,
) -> torch.Tensor:
    """Attention probabilities of one decode query over ``n_keys`` cached keys.

    Args:
        q: ``[num_query_heads, head_size]`` -- one decode token, post-RoPE.
        k_view: canonical ``(blocks, block_size, kv_heads, head_size)`` view.
        block_table: 1-D physical block ids for this request, on device.
        n_keys: how many leading cached positions to score (the prompt length).
        num_query_heads: query heads for this rank; ``//kv_heads`` gives the
            GQA fan-out.
        scratch: optional ``[num_query_heads, >=n_keys]`` float32 buffer.

    Returns:
        ``[num_query_heads, n_keys]`` float32, each row summing to 1.
    """
    _, block_size, num_kv_heads, head_size = k_view.shape
    if num_query_heads % num_kv_heads:
        raise ValueError(f"{num_query_heads} query heads not divisible by {num_kv_heads} kv")
    group = num_query_heads // num_kv_heads
    scale = 1.0 / math.sqrt(head_size)
    n_blocks = (n_keys + block_size - 1) // block_size

    if scratch is None or scratch.shape[1] < n_keys:
        scratch = torch.empty(num_query_heads, n_keys, dtype=torch.float32, device=q.device)
    logits = scratch[:, :n_keys]
    # ZEROED, not left as whatever was in memory. Every position in
    # [0, n_keys) should be written by the grid below, but if any is not, an
    # uninitialised buffer turns that into NaN after the softmax -- which
    # propagates through the whole reduction and is far harder to trace than
    # an obviously-wrong zero. Costs one pass over a [heads, n_keys] f32.
    logits.zero_()

    # Tokens per inner tile. 128 keeps the [TILE, PAD_D] tile small enough to
    # stay in registers at any head size the connector supports, and divides
    # the common block sizes (16, 32, 64, 128) in a single iteration.
    tile = min(128, triton.next_power_of_2(block_size))

    if not HAS_TRITON:
        return decode_attention_torch(
            q, k_view, block_table, n_keys, num_query_heads=num_query_heads
        )

    _qk_kernel[(num_query_heads, n_blocks)](
        q, k_view, block_table, logits,
        n_keys, scale,
        *q.stride(), *k_view.stride(), logits.stride(0),
        BLOCK_SIZE=block_size,
        GROUP=group,
        HEAD_SIZE=head_size,
        PAD_D=triton.next_power_of_2(head_size),
        TILE=tile,
        N_TILES=(block_size + tile - 1) // tile,
    )
    # Softmax over the sequence: the row is its own normaliser, so FlashAttention's
    # LSE is not needed.
    return torch.softmax(logits, dim=-1)


def decode_attention_torch(
    q: torch.Tensor,
    k_view: torch.Tensor,
    block_table: torch.Tensor,
    n_keys: int,
    *,
    num_query_heads: int,
) -> torch.Tensor:
    """Pure-torch reference for :func:`decode_attention`."""
    block_size = k_view.shape[1]
    num_kv_heads = k_view.shape[2]
    group = num_query_heads // num_kv_heads
    head_size = k_view.shape[3]

    tok = torch.arange(n_keys, device=q.device)
    blk = block_table.to(torch.long)[tok // block_size]
    off = tok % block_size
    k = k_view[blk, off].float()                       # (n_keys, kv_heads, D)
    k = k.repeat_interleave(group, dim=1)              # (n_keys, q_heads, D)
    logits = torch.einsum("hd,nhd->hn", q.float(), k) / math.sqrt(head_size)
    return torch.softmax(logits, dim=-1)
