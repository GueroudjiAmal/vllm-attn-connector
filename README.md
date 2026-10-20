# vllm-attn-connector

Captures **exact per-decode-step attention over the prompt** from a running vLLM
engine and emits it as **Flowcept provenance**.

For each decode step `t` and prompt token `j` it records

```
a[t, j] = softmax( q_t · K_promptᵀ / √d )[j]
```

reduced over layers and heads, per KV cache group. The result answers "which
prompt tokens did *this* generated token attend to", per token, rather than
"which prompt tokens mattered overall".

**No vLLM source is patched and no attention kernel is modified.**

## Repository layout

```
src/vllm_attn_connector/
    connector.py    KVConnector: recomputes q.Kt, reduces, selects, emits
    probe.py        attention-backend override that copies the decode query
    kernels.py      Triton q.Kt against the paged cache, plus a torch reference
    layout.py       KV cache layout resolution across vLLM backends
    store.py        SafeTensors writer, and the loader consumers use
tests/
    test_aggregations.py   pure torch, no GPU or vLLM needed
    e2e_smoke.py           real engine, asserts 43-45 properties of the output
    e2e_example.py         real engine, a 5-round chain with declared ranges
    bio_example.csv        the chain
    README.md              what each suite covers, and what is not covered
```

> This repository previously held `vllm-kvnorm`, a connector that scored
> cache-only KV norms. That approach was retired: statistics computable from the
> cache alone are request-invariant, so they cannot explain request-specific
> behaviour. The query is the missing ingredient, and it is never cached — hence
> the backend probe here. The kvnorm history remains in this repository's log.

## How it works

The scores are **recomputed, not extracted**. FlashAttention tiles the softmax
and discards the intermediate scores, so there is nothing to read back. But `K`
persists in the paged cache, and a single decode row is a matvec — cheap enough
to redo on a side stream.

Queries are the missing ingredient: they are never cached. A small backend
override copies them during the forward pass; everything else happens in the
connector.

```
step N
  bind_connector_metadata()   connector learns this step's requests
  start_load_kv()             emit anything that finished last step
  <forward>
      impl.forward(...)       PROBE: if max_query_len==1, copy q   [blocking, ~16 KB/layer]
                              then super().forward() unchanged
  wait_for_save()             CONNECTOR: q·Kᵀ, softmax, reduce      [overlapped, side stream]
```

| stage | cost | blocking |
|---|---|---|
| probe copies `q` | `heads × head_size × n_reqs` per layer | yes, sub-microsecond |
| `q·Kᵀ`, softmax, reduce | one matvec per request per layer | no — side stream |
| `event.synchronize()` at finish | one event per request | yes, once |

Two implementation notes that are easy to get wrong if you fork this:

**The probe overrides the backend impl, not the `Attention` module.**
`Attention.forward` is traced through by `torch.compile`; only the custom op
`unified_attention_with_output` survives as a runtime node, and it dispatches
`self.impl.forward(...)` dynamically. An `nn.Module` forward hook would not fire
at all.

**The query copy targets a persistent buffer, because CUDA graphs replay
kernels and not Python.** Being on the traced path is necessary but not
sufficient: a graph records the kernels a trace emits once, and replay re-runs
only those kernels. Any Python in the probe — a dict write, an allocation —
executes at capture time and never again. What does re-run is the copy itself,
so the probe allocates one buffer per layer, sized to `max_num_seqs`, before
capture, and every graph records a copy into a prefix of it. After any replay
the first `m` rows hold the current step's queries.

This matters because CUDA graphs are on by default in `vllm serve`. Getting it
wrong is silent rather than loud: the connector still emits structurally valid
records, they are just all zero, and even mass conservation still "passes"
because `0 + 1 == 1`. The e2e therefore runs with graphs **on** by default and
asserts that `attn_sum` totals one softmax unit per decode step.

**Decode only.** In decode each request contributes exactly one query token, so
batch row *i* is request slot *i* — no `query_start_loc` parsing and no
token-to-request mapping is needed. Prefill is skipped; see
[Limitations](#limitations).

## Install

```bash
ROOT=$(cd .. && pwd)
export PYTHONPATH="$ROOT/vllm-attn-connector/src:$ROOT/flowcept/src"
export FLOWCEPT_SETTINGS_PATH="$ROOT/flowcept/agent_sandbox/settings.yaml"
```

Flowcept must come from a checkout that contains
`flowceptor/adapters/vllm/`.

Requires a CUDA device and Triton. Tested against vLLM 0.27.x and 0.28.0 with
`VLLM_ENABLE_V1_MULTIPROCESSING=0`; with multiprocessing enabled the emitted
records leave the process and Flowcept needs a reachable message queue.

### Cache layouts

Uniform caches and **hybrid** ones both work. A hybrid model interleaves full
attention with another mechanism — gated delta net in Qwen3.5 — so only some
layers have a scorable KV cache, and vLLM gives each kind its own cache group
with its own block table. The connector scores the full-attention group and
takes its block ids from that group, not from group 0.

Two consequences worth knowing before adding a model:

- **The paged block size need not be a power of two.** vLLM sizes a hybrid
  model's attention page to match the other mechanism's state page; Qwen3.5
  uses 1056 tokens. The kernel walks each block in fixed 128-token tiles for
  this reason, so block size is unconstrained.
- **Multimodal configs nest the language model's shape** under
  `text_config`. Head counts are read from `hf_text_config`; if that ever
  fails the fallback warns rather than guessing, because guessing wrong means
  scoring a fraction of the heads with nothing in the log.

## Quickstart

```python
import vllm_attn_connector
# MUST precede engine construction: _cached_get_attn_backend is @cache'd, so the
# backend class is resolved once and never re-read.
assert vllm_attn_connector.install_probe()

from flowcept import Flowcept
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

with Flowcept("vllm", workflow_id="my-run", workflow_name="my_run"):
    llm = LLM(
        model="Qwen/Qwen3-4B-Instruct-2507",
        kv_transfer_config=KVTransferConfig(
            kv_connector="AttnConnector",
            kv_connector_module_path="vllm_attn_connector",
            kv_role="kv_producer",
            kv_connector_extra_config={"workflow_id": "my-run",
                                       "out_dir": "/scratch/prov"},
        ),
    )
    llm.generate(["..."], SamplingParams(max_tokens=64))
```

Each finished request writes one file under `out_dir` and one Flowcept task
pointing at it. To read the statistics back:

```python
from vllm_attn_connector import load_provenance

task = ...                                  # a Flowcept task record
arrays = load_provenance(task)
arrays["val_all_max"]                       # (decode steps, k)
```

Use `kv_role="kv_producer"`, not `"kv_both"` — this connector never loads KV,
and declaring it a consumer makes the scheduler defer block frees.

## Configuration

All keys go in `kv_connector_extra_config`.

| key | default | meaning |
|---|---|---|
| `workflow_id` | — | parent workflow to attach records to |
| `out_dir` | `./attention_provenance` | where the statistics files are written |
| `checksum` | `true` | record a sha256 of each file; costs a pass over the bytes |
| `max_steps` | `0` | decode steps recorded per request; `0` = every one |
| `top_pct` | `10` | percent of positions kept within each segment |
| `chunk_size` | `32` | segment width when no ranges are declared; `0` = one segment |
| `layer_stats` | `false` | opt-in per-head diagnostics; roughly +50% cost |

By default **every decode token is recorded**. Step-indexed buffers start at 64
rows and double as needed, so unbounded capture costs no more than a bounded one
for short generations, and long generations are not silently truncated. Set a
positive `max_steps` to cap memory explicitly; steps beyond the cap are counted
in `decode_steps_dropped` rather than lost quietly.

```json
{
  "kv_connector": "AttnConnector",
  "kv_connector_module_path": "vllm_attn_connector",
  "kv_role": "kv_producer",
  "kv_connector_extra_config": {
      "workflow_id": "my-run",
      "out_dir": "/scratch/prov",
      "top_pct": 10,
      "chunk_size": 32
  }
}
```

### Segments: fixed or variable

Storing the full `[decode_steps, prompt_tokens]` matrix does not scale — it is
linear in prompt length and grows without bound on long contexts. Instead the
prompt is **partitioned into segments** and the strongest `top_pct` percent of
positions is kept *within each one*.

Selecting per segment rather than over the whole prompt is the point: a global
top-X% concentrates wherever the distribution happens to peak and can leave
entire regions with no stored entry at all, which makes those regions invisible
downstream. Per-segment selection guarantees every region is represented, with
a floor of one entry per segment however short it is.

Segments come from one of two places, and **the output is identical either way**:

**Fixed** (default) — uniform blocks of `chunk_size` tokens. Use when the prompt
has no structure you can declare.

**Variable** — ranges you declare per request, one per document, tool output,
retrieved chunk, or whatever unit you want statistics for:

```python
SamplingParams(
    max_tokens=256,
    extra_args={"kv_transfer_params": {"ranges": [[131, 496], [496, 548], [578, 765]]}},
)
```

Ranges are clamped to the prompt, sorted, and the **gaps between them become
segments too**, so the partition stays complete and `topk_residual` stays exact.
Overlaps are resolved by truncation. Malformed input is ignored with a warning
rather than raised — provenance capture must never fail a generation request.

Whichever mode produced a record, the partition is emitted as `segments`, so a
consumer runs the same per-segment aggregation without knowing or caring which
was used. `metadata.segment_mode` says which it was.

Entries kept per step is `sum(max(1, round(top_pct/100 × len(seg))))` over
segments. Note the round-and-floor: with `top_pct` small and segments short, the
floor of 1 dominates and the effective rate is higher than `top_pct`.

**Full capture** — `chunk_size=1, top_pct=100` makes every position its own
segment and keeps all of them. That is the replacement for the dense mode this
connector used to have: same information, same field names, no separate code
path.

## What is recorded

The statistics do not travel inside the provenance record. They are written to
one **SafeTensors** file per (request, KV cache group), and the record carries a
reference. A record is then ~1.6 KB whatever the prompt length; inline, a
128k-context request serialises to hundreds of megabytes of JSON, past what a
message queue accepts, and building those Python lists costs a device sync.

**On the workflow, once per run** — `attention_config`: the settings that
produced the files (`top_pct`, `chunk_size`, `max_steps`, `head_code`,
`num_query_heads`, `num_layers_scored`, `selected_on`, ...). It is constant for
the engine, so repeating it per task would be duplication that can drift. The
model and tokenizer are already alongside it in `conf`.

**On each task** — `used` (request id, prompt and decode token counts) and
`attention_stats`:

```json
"attention_stats": {
  "uri": "file:///scratch/prov/<workflow>/<request>_g0.safetensors",
  "format": "safetensors", "bytes": 110652, "sha256": "...",
  "kv_cache_group_id": 0, "written_by_tp_rank": 0,
  "segment_mode": "variable",
  "decode_steps_dropped": 0, "decode_steps_nonfinite": 0, "restarts": 0,
  "tensors": {"attn_sum": {"shape": [3540], "dtype": "float32"}, ...}
}
```

`segment_mode` and the three health counters stay per-task on purpose: the
first depends on whether *that* request declared ranges, and the others are how
a single untrustworthy record is identified.

### Tensors in the file

Write `G` for decode steps recorded, `T` for prefill tokens scored, `k` for
entries kept per step.

| tensor | shape | meaning |
|---|---|---|
| `topk_pos` | `[G, k]` int32 | retained prefill positions, ascending; `-1` pads the ragged final segment |
| `val_all_max` | `[G, k]` f32 | attention at those positions, **max** over all (layer, head) |
| `val_all_avg` | `[G, k]` f32 | attention at those positions, **mean** over all (layer, head) |
| `topk_head` | `[G, k]` int32 | which head supplied the max; see `head_code` |
| `topk_residual` | `[G, 1]` f32 | mean-aggregation mass that selection discarded |
| `segments` | `[n_seg, 3]` int32 | the partition, as `(lo, hi, keep)` |
| `attn_sum` | `[T]` f32 | column sum of the mean row, over **all** positions |
| `attn_peak` | `[T]` f32 | max over (layer, head, decode step), over **all** positions |
| `prompt_token_ids` | `[T]` int32 | the prompt; decode it with the tokenizer on the workflow |

`segments` covers `[0, T)` with no gaps or overlaps, so every retained position
falls in exactly one segment and a `searchsorted` on the segment starts maps
positions to segments.

`attn_sum` and `attn_peak` cover every position, including those selection
dropped, so whole-prompt totals remain available.

With `layer_stats` on, the file also carries the per-head diagnostics
(`head_argmax_pos`, `head_entropy`, `wins_first_step`, `layer_maxsum`, ...).

### Reading it back

```python
from vllm_attn_connector import load_provenance, open_lazy

arrays = load_provenance(task_record)      # dict of numpy arrays
arrays["val_all_max"].shape                # (G, k)

with open_lazy(task_record) as f:          # one tensor out of a large file
    segs = f.get_tensor("segments")
```


The file also carries a small string header (request id, segment mode,
`top_pct`, `head_code`), so it stays interpretable if it is separated from the
provenance store that references it.

### Why both a max and a mean

They answer different questions and neither substitutes for the other.

`val_all_max` is what selection runs on and what tends to be useful for
attributing a generated token to a prompt span. A mean over all (layer, head)
pairs dilutes: if a small number of heads carry the retrieval behaviour,
averaging them against hundreds that do not will bury the signal.

`val_all_avg` is a true probability distribution over positions, which is what
makes `topk_residual` meaningful — it accounts for exactly the mass selection
threw away, so `sum(val_all_avg) + topk_residual == 1` per step. A max cannot
express that, because maxima do not sum to anything.

`topk_head` is free: the max reduction has to identify the winning head anyway.

### Position 0 is never selected

The first prompt token is the attention sink — in decoder-only LLMs it absorbs a
large, roughly content-independent share of attention
([arXiv:2309.17453](https://arxiv.org/abs/2309.17453)). Letting it win a slot
would spend one of the `per_chunk` entries on a token that carries no
information about what the model is doing.

Its mass remains fully visible in `attn_sum` and `attn_peak`, which cover all
positions. If you are consuming `topk_pos` you do not need to mask the sink
yourself: it is never there.

## Output size

The provenance record is **flat** — a descriptor, not the data — so it does not
grow with the prompt. The file does:

```
floats = k · G · 4      (positions + 2 values + head id)
       +     G          (residual)
       +   2·T          (attn_sum, attn_peak)
       +     T          (prompt_token_ids)
       + 3·n_seg        (the partition)
```

as float32/int32, not text. **Size does not depend on the number of heads,
layers, or head_size** — those are reduced before anything is stored; they drive
*compute*, not output.

| variable | meaning | effect on file size |
|---|---|---|
| `T` | prefill tokens scored | `3T`, from `attn_sum`, `attn_peak`, `prompt_token_ids` |
| `G` | decode steps recorded: all of them, or `max_steps` if capped | linear |
| `k` | entries kept per step | linear |
| `n_groups` | KV cache groups: 1 uniform, 2 for some hybrid models | one file each |
| `H`, `L`, `head_size` | heads, layers, head dim | **none** |

Measured on a five-round chain, 953 → 3540 prompt tokens:

| round | T | G | k | record | file |
|---|---|---|---|---|---|
| 1 | 953 | 8 | 95 | 1.6 KB | 23 KB |
| 3 | 2308 | 11 | 231 | 1.6 KB | 67 KB |
| 5 | 3540 | 12 | 353 | 1.6 KB | 108 KB |

With the default `max_steps=0`, `G` is the full generation length, so the file
grows with how much the model produces. A positive `max_steps` bounds it. `k` is
under your direct control via `top_pct` and the segment widths, which is what
keeps long prompts affordable.

Files are written to `out_dir/<workflow_id>/<request_id>_g<group>.safetensors`
via a temporary name and an atomic rename, so a reader tailing the directory
never sees a partial file. A write failure is logged and reported as
`attention_stats.error` rather than raised — provenance capture must not be able
to fail a generation request.

## Performance

Cost is dominated by recomputing `q·Kᵀ`, which re-reads the prompt's keys from
the paged cache once per decode step per layer. It is memory-bandwidth bound,
so it scales with `prompt_tokens × layers × heads × head_size` and is largely
insensitive to what you do with the result afterwards.

Practical consequences:

- **Selection does not reduce compute**, only output size. Keeping 10% of
  positions costs the same as keeping all of them. Choose `top_pct` and the
  segmentation for storage and for the granularity you want statistics at, not
  for speed.
- **Longer prompts cost proportionally more**, since the whole prompt's `K` is
  re-read every step.
- **Cost is per decode step**, so it scales with generation length. `max_steps`
  caps that if you need a ceiling; by default there is none.
- **Per-head detail is expensive.** Anything that has to carry per-head state
  through the reduction — such as `layer_stats` — costs substantially more than
  the reduced statistics, because the reduction is what keeps the inner loop
  cheap.

On a single consumer GPU with a 4B model and prompts of a few thousand tokens,
end-to-end generation slowdown is high single digit percent, and recording
every decode token rather than a capped prefix does not measurably change it.
Measure on your own workload before budgeting: the ratio depends on prompt
length, generation length and how much headroom the model leaves on your
device.

Two measurement pitfalls, both of which produced wrong numbers during
development: interleave capture and no-capture runs **in one session**, since
GPU clock and thermal state drift between sessions by more than the effect
being measured; and discard the first run of a session, which pays warmup the
others do not.

## Validation

```bash
# no GPU, no vLLM: streaming reduction and segment planning vs references
pytest tests/test_aggregations.py

# end to end against a real engine
python tests/e2e_smoke.py                                 # CUDA graphs ON, as served
python tests/e2e_smoke.py --enforce-eager                 # graphs off
python tests/e2e_smoke.py --ranges                        # variable segments
python tests/e2e_smoke.py --chunk-size 1 --top-pct 100    # full capture
python tests/e2e_smoke.py --top-pct 25 --chunk-size 64
python tests/e2e_example.py                               # 5-round chain, declared ranges
```

See [`tests/README.md`](tests/README.md) for what each covers, and for what is
*not* covered — preemption and tensor parallelism both lack tests.

The e2e asserts 43-45 properties depending on the configuration. The
load-bearing ones:

- **`attn_sum` totals one softmax unit per decode step.** Attention is a
  distribution, so this is the check that the scores were actually written. It
  is asserted separately from mass conservation because an all-zero capture
  satisfies the latter trivially.
- **`sum(val_all_avg) + topk_residual == 1`** at every step, so the output is a
  genuine probability distribution with the discarded mass accounted for
  exactly, not a plausible-looking artifact.
- **The partition is complete**: segments cover the prompt with no gaps or
  overlaps, and every declared range appears verbatim as a segment.
- **The manifest matches the file**: the tensor names and shapes recorded in
  `attention_stats` are exactly what reading the file back produces. A
  descriptor that disagrees with its data is worse than no descriptor.
- **The attention sink is reproduced** at two to three orders of magnitude above
  the median position. Recovering a known property of the model is evidence the
  recomputation is reading real keys.
- **Attention changes between decode steps.** Under selection this shows up as
  turnover in the retained position set (adjacent-step Jaccard well below 1);
  under full capture it is measured as total-variation distance between
  consecutive steps. If attention did not move, per-step capture would be
  redundant with `attn_sum` and this package would have no purpose.
- **Every segment is represented in every step**, the invariant that
  per-segment selection exists to provide.

The Triton kernel matches a pure-torch reference to ~1e-08 across full, partial
and single-key sequences.

## Limitations

- **Prefill attention is not captured.** Only decode queries are scored;
  attention *between* prompt tokens during prefill is never computed. If the
  content behind an answer was assembled into a late prompt position during
  prefill, decode attention points at that position rather than at the original
  source. Capturing it would require streaming during prefill, since prefill
  queries are not cached either.
- **Attention magnitude is not signed.** A head attending strongly to a token
  may be suppressing it as easily as using it. High attention means "this token
  was consulted", not "this token was used affirmatively".
- **Tensor parallelism is handled but untested.** The owner is reduced
  alongside the value, so `topk_head` is a global head index at any `tp_size`
  and always names the head that produced `val_all_max`. That costs one
  all-gather of `(value, owner)` per step on top of the two all-reduces. Only
  rank 0 writes, so there is one file per request regardless of `tp_size`. The
  arithmetic is unit-tested against a dense argmax at world sizes 2, 4 and 8;
  the path has never run on more than one GPU.
- **Reduced over layers and heads.** The output cannot tell you *which* layer
  produced a score, only which head supplied the maximum. `layer_stats` exposes
  per-head detail at significant cost.
- **Segments are frozen at the first decode step** and derived from the prompt
  length settled at that point. Ranges are read from the request's first
  appearance only; they describe the prompt, which does not change.
- **Very uneven ranges cost memory.** Selection pads segments to the widest one,
  so one range far larger than the rest inflates a `[n_seg, max_width]`
  scratch buffer. Even-ish segments, or a `chunk_size` grid, avoid this.
- **Scoring cannot currently be restricted to a subset of layers.** Since cost
  is dominated by re-reading `K` per layer, this is the most promising available
  speedup and is not yet implemented.
- **Sliding-window and other non-full attention groups** are scored against the
  positions actually in cache for that group; `scored_attention` and
  `attention_window` in the metadata record which regime applied.
- **Preemption drops partial records.** If a request is preempted and restarted,
  steps recorded before the restart are discarded rather than stitched, and
  `restarts` is incremented.
- **`layout.py` is vendored**, duplicated with `vllm-kvnorm`. If a third package
  needs it, extract a shared dependency instead of copying again.
- **A dead capture is not self-announcing.** If the probe never fires, or the
  cache is read with the wrong block table, every record is still written and
  is still structurally valid: the manifest matches, the declared ranges are
  present, and mass conservation holds because `0 + 1 == 1`. Two checks do
  catch it, and a consumer should run both — that any retained position is
  `> 0`, and that `attn_sum` totals one softmax unit per recorded decode step.
  `tests/e2e_smoke.py` asserts the second; nothing asserts either at runtime.
