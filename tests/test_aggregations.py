#!/usr/bin/env python3
"""Unit checks for the per-position head reduction, no GPU or vLLM needed.

The connector reduces each decode step's [L, H, T] attention to two [T] vectors
plus a winner id, accumulating layer by layer so the full tensor is never
materialised. These tests replicate that streaming reduction and compare it
against the obvious dense computation.

Also pins the two facts that decided the emitted schema:

  * a max over the top K heads equals the max over all heads identically, so
    such a column could never carry information;
  * a mean over the top K heads *is* distinct from both emitted aggregations --
    it is a real statistic that was left out on cost grounds, not because it is
    redundant. The test keeps that distinction checkable.

Run: python experiments/vllm-attn-connector/test_aggregations.py
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import torch

# Loaded by PATH, not as `vllm_attn_connector.kernels`: importing the package
# runs its __init__, which imports connector.py, which imports vLLM. This file
# is the suite that must run with no GPU and no vLLM, and kernels.py itself has
# no vLLM dependency -- only the package entry point does.
_KERNELS = (pathlib.Path(__file__).resolve().parent.parent
            / "src" / "vllm_attn_connector" / "kernels.py")
_spec = importlib.util.spec_from_file_location("_attn_kernels", _KERNELS)
_kernels = importlib.util.module_from_spec(_spec)
sys.modules["_attn_kernels"] = _kernels
_spec.loader.exec_module(_kernels)
decode_attention_torch = _kernels.decode_attention_torch


def streaming(layers: list[torch.Tensor]):
    """What score_step does: one layer at a time, O(T) state."""
    T = layers[0].shape[1]
    acc = torch.zeros(T, dtype=torch.float32)
    run_max = torch.full((T,), float("-inf"), dtype=torch.float32)
    owner = torch.full((T,), -1, dtype=torch.int64)
    for li, probs in enumerate(layers):
        acc += probs.mean(0)
        lmax, hidx = probs.max(0)
        better = lmax > run_max
        owner = torch.where(better, li * probs.shape[0] + hidx, owner)
        run_max = torch.maximum(run_max, lmax)
    return {"val_all_max": run_max,
            "val_all_avg": acc / len(layers),
            "topk_head": owner}


def dense(layers: list[torch.Tensor]):
    """The definition, with the whole [L*H, T] tensor in memory."""
    a = torch.cat(layers, 0)
    v, i = a.max(0)
    return {"val_all_max": v, "val_all_avg": a.mean(0), "topk_head": i}


def case(name, L, H, T, mk):
    layers = [mk(H, T) for _ in range(L)]
    got, want = streaming(layers), dense(layers)
    worst = 0.0
    for key in ("val_all_max", "val_all_avg"):
        d = (got[key] - want[key]).abs().max().item()
        worst = max(worst, d)
        assert d < 1e-6, f"{name}/{key}: streaming != dense, max diff {d:g}"
    # The winner is compared by the value it carries: ties are broken
    # arbitrarily and a tied swap is not an error.
    a = torch.cat(layers, 0)
    gv = torch.gather(a, 0, got["topk_head"].clamp_min(0).unsqueeze(0))[0]
    dv = (gv - want["val_all_max"]).abs().max().item()
    assert dv < 1e-6, f"{name}/topk_head: winner does not carry the max ({dv:g})"
    exact = (got["topk_head"] == want["topk_head"]).float().mean().item()
    print(f"  ok  {name:<34} L={L:<3} H={H:<4} T={T:<5} "
          f"max|diff|={worst:.2e}  winner id exact {exact:.0%}")


def test_reduction() -> None:
    """Streaming layer-by-layer reduction must equal the dense definition."""
    torch.manual_seed(0)
    rnd = lambda H, T: torch.rand(H, T)
    print("streaming layer-by-layer reduction == dense reduction")
    case("uniform random", 36, 32, 512, rnd)
    case("single layer", 1, 32, 128, rnd)
    case("single head", 8, 1, 64, rnd)
    case("one hot per column", 12, 8, 64,
         lambda H, T: torch.nn.functional.one_hot(
             torch.randint(0, H, (T,)), H).T.float())
    case("softmax rows (realistic)", 36, 32, 384,
         lambda H, T: torch.softmax(torch.randn(H, T) * 3, dim=1))

    print("\nwhy there is no val_topk_max column")
    for kh in (1, 2, 3, 8, 64):
        a = torch.rand(1152, 97)
        assert torch.equal(a.topk(kh, dim=0).values[0], a.max(0).values), kh
        print(f"  ok  K={kh:<4} max over top-K == max over all heads, all 97 positions")

    print("\nwhy mean-over-top-K is distinct, yet not emitted")
    a = torch.rand(1152, 97)
    lo, mid, hi = a.mean(0), a.topk(3, dim=0).values.mean(0), a.max(0).values
    assert (lo <= mid + 1e-6).all() and (mid <= hi + 1e-6).all()
    strict = ((lo < mid - 1e-6) & (mid < hi - 1e-6)).float().mean().item()
    print(f"  ok  all_avg <= topk_avg <= all_max, strict at {strict:.0%} of positions")
    print("      -> a real third statistic, but it is only interpretable with the")
    print("         K head identities alongside it, and carrying per-head state")
    print("         through the reduction is what makes capture expensive.")
    print("         Only the winner (topk_head) is emitted; it is free.")


def test_owner_reduction() -> None:
    """`_resolve_owner` must name the head a single unsharded run would.

    Calls the connector's own function -- not a copy of the formula -- with
    hand-built gathered tensors, so a change to the real arithmetic fails here.
    The collectives around it need a process group and more than one GPU; this
    index arithmetic is the part that can silently be wrong.
    """
    resolve = _load_segment_fns()["_resolve_owner"]
    torch.manual_seed(0)
    for world, hl, L, T in ((4, 8, 3, 6), (2, 16, 2, 5), (8, 4, 5, 9)):
        full = torch.rand(L, world * hl, T)          # ground truth: every head
        V, O = [], []
        for r in range(world):
            sl = full[:, r * hl:(r + 1) * hl].reshape(-1, T)
            v, h = sl.max(0)                         # this rank's max, local code
            V.append(v)
            O.append(h)
        val, own = resolve(torch.stack(V), torch.stack(O).long(), hl)

        want_v, want_c = full.reshape(-1, T).max(0)
        assert torch.allclose(val, want_v), f"world={world}: value mismatch"
        assert torch.equal(own, want_c), f"world={world}: global head mismatch"
        print(f"  ok  world={world:<2} {hl:>2} heads/rank, {L} layers -> "
              f"{world*hl:>3} global heads, T={T}")

    # -1 marks a position no head has claimed; it must pass through untouched
    V = torch.tensor([[0.5, 0.1], [0.2, 0.9]])
    O = torch.tensor([[-1, 3], [2, -1]])
    _, own = resolve(V, O, 4)
    assert own[0].item() == -1, "sentinel from the winning rank was re-encoded"
    assert own[1].item() == -1
    print("  ok  the -1 no-owner sentinel is preserved")


def main() -> int:
    test_reduction()
    print()
    test_segments()
    print("\ntensor-parallel owner reduction")
    test_owner_reduction()
    print("\nall checks passed")
    return 0


# --------------------------------------------------------------------------
# Segment planning and selection. Imported from the connector source directly:
# the module needs vLLM to import, these two functions do not.
# --------------------------------------------------------------------------

def _load_segment_fns():
    import ast as _ast
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[1]
           / "src/vllm_attn_connector/connector.py").read_text()
    tree = _ast.parse(src)
    want = {"_segment_plan", "_segment_index", "_segmented_topk", "_resolve_owner"}
    mod = _ast.Module(body=[n for n in tree.body
                            if isinstance(n, _ast.FunctionDef) and n.name in want],
                      type_ignores=[])
    ns = {"torch": torch, "NO_ENTRY": -1}
    exec(compile(mod, "<connector>", "exec"), ns)
    return ns


def test_block_sizes_not_powers_of_two() -> None:
    """The reference scores every cached position at any paged block size.

    Triton's ``tl.arange`` needs a power-of-two extent, but vLLM chooses the
    paged block size per model and backend and does not guarantee one: a
    Qwen3.5 MoE (head_dim 256, 2 KV heads) picked a value that made the kernel
    fail to compile with "arange's range must be a power of 2". The kernel now
    iterates a padded extent and masks the tail, exactly as it already did for
    head_size via PAD_D.

    This test covers the torch reference rather than the Triton kernel, since
    it has to run without a GPU -- but it pins the arithmetic the padded kernel
    has to reproduce: every position in [0, n_keys) is scored exactly once, and
    none of the padding leaks in.
    """
    import math

    def naive(q, kv, bt, n_keys, nqh):
        bs, nkv, hd = kv.shape[1], kv.shape[2], kv.shape[3]
        group = nqh // nkv
        out = torch.zeros(nqh, n_keys)
        for pos in range(n_keys):
            blk, off = bt[pos // bs].item(), pos % bs
            for h in range(nqh):
                out[h, pos] = (q[h].float() * kv[blk, off, h // group].float()
                               ).sum() / math.sqrt(hd)
        return torch.softmax(out, dim=-1)

    torch.manual_seed(0)
    for block_size in (16, 12, 24, 48, 80, 100):
        nqh, nkv, head_dim, n_blocks = 8, 2, 64, 6
        q = torch.randn(nqh, head_dim)
        kv = torch.randn(n_blocks, block_size, nkv, head_dim)
        bt = torch.arange(n_blocks)
        # a partial final block, so the n_keys mask is exercised too
        n_keys = min(block_size * n_blocks, block_size * 3 + 5)
        got = decode_attention_torch(q, kv, bt, n_keys, num_query_heads=nqh)
        want = naive(q, kv, bt, n_keys, nqh)
        assert got.shape == (nqh, n_keys), (block_size, got.shape)
        assert torch.allclose(got, want, atol=1e-5), (
            f"block_size={block_size}: max diff "
            f"{(got - want).abs().max().item():.2e}")
        assert torch.allclose(got.sum(-1), torch.ones(nqh), atol=1e-5)

    # and the padding arithmetic itself: n < BLOCK_SIZE must leave no gaps
    for block_size in (12, 24, 80):
        pad_n = 1 << (block_size - 1).bit_length()
        covered = sorted({b * block_size + n
                          for b in range(3) for n in range(pad_n)
                          if n < block_size and b * block_size + n < block_size * 3})
        assert covered == list(range(block_size * 3)), block_size


def test_segments() -> None:
    ns = _load_segment_fns()
    plan_fn, index_fn, topk_fn = (ns["_segment_plan"], ns["_segment_index"],
                                  ns["_segmented_topk"])
    torch.manual_seed(0)

    def run(n_keys, chunk, pct, ranges, label):
        plan = plan_fn(n_keys, chunk, pct, ranges)
        idx, live, want, total = index_fn(plan, "cpu")
        kmax = max(k for _, _, k in plan)
        row = torch.rand(n_keys)
        pos, val = topk_fn(row, idx, live, want, total, kmax)
        p = [int(x) for x in pos.tolist() if x >= 0]

        # the partition is complete, ordered and non-overlapping
        assert plan[0][0] == 0 and plan[-1][1] == n_keys, f"{label}: not covering"
        for (a, b, _), (c, d, _) in zip(plan, plan[1:]):
            assert b == c, f"{label}: gap/overlap at {b}!={c}"
        # positions are legal, unique, ascending, never the sink
        assert all(0 < x < n_keys for x in p), f"{label}: out of range or sink"
        assert len(set(p)) == len(p), f"{label}: duplicates"
        assert p == sorted(p), f"{label}: not ascending"
        # each segment got exactly its quota (minus the sink it cannot use)
        for lo, hi, keepn in plan:
            got = sum(1 for x in p if lo <= x < hi)
            cap = min(keepn, (hi - lo) - (1 if lo == 0 else 0))
            assert got == cap, f"{label}: segment [{lo},{hi}) got {got}, wanted {cap}"
        # values match the row at the positions claimed
        for x, v in zip(pos.tolist(), val.tolist()):
            if x >= 0:
                assert abs(row[x].item() - v) < 1e-6, f"{label}: value mismatch"
        print(f"  ok  {label:<44} segs={len(plan):<4} k={total}")
        return plan

    print("segment planning and selection")
    run(200, 32, 10, None, "fixed, chunk 32, 10%")
    run(200, 1, 100, None, "fixed, chunk 1, 100% (every position)")
    run(200, 512, 10, None, "fixed, chunk wider than prompt")
    run(37, 32, 10, None, "fixed, ragged final chunk")
    run(200, 32, 10, [(10, 60), (80, 95), (120, 190)], "variable, 3 ranges + gaps")
    run(200, 32, 10, [(0, 200)], "variable, one range spanning all")
    run(200, 32, 10, [(0, 50), (50, 200)], "variable, adjacent, no gaps")
    run(200, 32, 10, [(150, 400)], "variable, range past the prompt end")
    run(200, 32, 10, [(20, 80), (50, 120)], "variable, overlapping (truncated)")
    run(200, 32, 0, None, "top_pct=0 -> floor of 1 per segment")

    # fixed mode must reproduce the uniform grid exactly
    plan = plan_fn(200, 32, 10, None)
    grid = [(i, min(i + 32, 200)) for i in range(0, 200, 32)]
    assert [(lo, hi) for lo, hi, _ in plan] == grid, "grid mismatch"
    print("  ok  fixed mode reproduces the uniform grid")

    # declaring ranges that happen to be the grid == fixed mode
    same = plan_fn(200, 32, 10, [(i, min(i + 32, 200)) for i in range(0, 200, 32)])
    assert same == plan, "declaring the grid as ranges should equal fixed mode"
    print("  ok  declaring the grid as ranges is identical to fixed mode")


if __name__ == "__main__":
    sys.exit(main())
