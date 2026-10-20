"""``AttnConnector`` -- exact per-decode-step attention over the prompt.

Two extension points, both public, neither patching vLLM:

    probe.py    backend override; copies the decode query        [in the forward]
    this file   KVConnector; recomputes q.K^T, accumulates, emits [side stream]

Per decode step ``t`` and prompt token ``j`` it records ``a[t, j]``, the exact
attention the generated token paid to that prompt token, aggregated over layers
and heads two ways -- mean and max.

Why per step, and not summed
----------------------------
A summed T-vector answers "which prompt tokens mattered overall". It cannot
answer "which prompt span produced *this* output token", which is the
attribution question, and the two are not interchangeable: a running sum
systematically misranks tokens whose importance recurs, and per-step selection
recovers the true top-k far better than cumulative selection
(arXiv:2510.02629, arXiv:2506.15969, Quest arXiv:2406.10774).

Evidence sometimes cited *for* using a sum measures union-vs-union containment
(Scissorhands, arXiv:2305.17118) or 128-step averages (SnapKV,
arXiv:2404.14469), never per-step overlap; work that measures turnover directly
finds adjacent-step Jaccard well below 1 (arXiv:2602.11162).

``attn_sum`` is emitted alongside anyway -- it is a free byproduct of the
accumulation and covers every position, including those selection drops.

Heads
-----
Both a max and a mean over (layer, head) are kept. The max is what selection
runs on and what survives when only a few heads carry the behaviour of
interest; a mean over hundreds of pairs flattens those. The mean is kept
because it is a true distribution, which is what makes ``topk_residual``
exact -- retained mean mass plus residual is 1 by construction.
"""

from __future__ import annotations

import math
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import TYPE_CHECKING, Any

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    KVConnectorWorkerMetadata,
    SupportsHMA,
)
from vllm.logger import init_logger

from .kernels import decode_attention
from .layout import LayerLayout, UnsupportedLayout, resolve_layer_layout
import numpy as np

from . import store
from .probe import REGISTRY, install

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.outputs import KVConnectorOutput
    from vllm.v1.request import Request

logger = init_logger("vllm.attn_connector")

_DEBUG = bool(os.environ.get("OPAL_ATTN_DEBUG"))

FLOAT_DECIMALS = 6
FULL_ATTENTION = "full_attention"
ACTIVITY = "decode_attention"
# 0 = record every decode token. Step-indexed buffers start at STEP_CHUNK rows
# and double as needed, so an unbounded default costs no more than a bounded one
# for short generations. Set a positive `max_steps` to cap memory explicitly;
# steps past the cap are counted in `decode_steps_dropped`, not silently lost.
DEFAULT_MAX_STEPS = 0
STEP_CHUNK = 64
DEFAULT_TOP_PCT = 10.0
DEFAULT_CHUNK = 32
DEFAULT_OUT_DIR = "./attention_provenance"
# Aggregations stored per retained position, both reduced over every
# (layer, head) pair. `val_all_max` is what selection runs on; `val_all_avg` is
# a true distribution, which is what makes `topk_residual` exact.
#
# The identity of the winning head (`topk_head`) comes along free -- the max
# reduction has to find it. Carrying more than the winner, e.g. the top K heads
# and their values, was tried and removed: per-head state through the reduction
# costs far more than the reduced statistics, which is the whole reason the
# inner loop is cheap.
#
# Note a max over the top K heads would be pointless regardless: the largest of
# the K largest values is the largest value, so it equals `val_all_max`
# identically for every K >= 1.
N_AGG = 2
AGG_FIELDS = ("val_all_max", "val_all_avg")
NO_ENTRY = -1


def _attn_spec(spec: Any) -> tuple[str, int | None]:
    """(kind, window) for a KV cache group. Duck-typed on the window attributes
    rather than isinstance-checked: FullAttentionSpec carries optional
    sliding_window/attention_chunk_size fields that merging can populate, and
    UniformTypeKVCacheSpecs wraps a dict of heterogeneous sub-specs."""
    inner = getattr(spec, "kv_cache_specs", None)
    if inner:
        seen = {_attn_spec(x) for x in inner.values()}
        if len(seen) == 1:
            return seen.pop()
        return "mixed(" + ",".join(sorted(k for k, _ in seen)) + ")", None
    for attr, label in (("sliding_window", "sliding_window"),
                        ("attention_chunk_size", "chunked_local")):
        w = getattr(spec, attr, None)
        if w is not None:
            return f"{label}({int(w)})", int(w)
    return FULL_ATTENTION, None


def _model_conf(vllm_config: "VllmConfig") -> dict[str, Any]:
    m = vllm_config.model_config
    conf = {k: getattr(m, k) for k in
            ("model", "tokenizer", "tokenizer_mode", "trust_remote_code", "max_model_len")}
    conf["dtype"] = str(m.dtype)
    conf["architectures"] = list(getattr(m, "architectures", []) or [])
    # From hf_text_config, not hf_config. A multimodal model nests the language
    # model's shape under `text_config`, and the top level has no
    # num_attention_heads at all -- Qwen3.5 reported 2 query heads against a
    # true 16 because every lookup missed and the fallback used num_kv_heads.
    _text = getattr(m, "hf_text_config", None) or getattr(m, "hf_config", None)
    for _k in ("num_attention_heads", "num_key_value_heads", "head_dim",
               "num_hidden_layers"):
        _v = getattr(_text, _k, None)
        if _v is not None:
            conf[_k] = int(_v)
    conf["tensor_parallel_size"] = vllm_config.parallel_config.tensor_parallel_size
    for k in ("revision", "tokenizer_revision", "served_model_name"):
        v = getattr(m, k, None)
        if v:
            conf[k] = v if isinstance(v, str) else list(v)
    return conf


@dataclass
class StepRequest:
    request_id: str
    new_block_ids: tuple[list[int], ...]
    num_computed_tokens: int
    num_scheduled_tokens: int
    prompt_token_ids: list[int] | None = None
    # Caller-declared prompt ranges, if any. Only present on the request's
    # first appearance, which is before scoring starts, so the plan sees them.
    ranges: list | None = None


@dataclass
class AttnMetadata(KVConnectorMetadata):
    scheduled: list[StepRequest] = field(default_factory=list)
    finished: list[str] = field(default_factory=list)


@dataclass
class AttnWorkerMeta(KVConnectorWorkerMetadata):
    emitted: set[str] = field(default_factory=set)

    def aggregate(self, other: "KVConnectorWorkerMetadata") -> "KVConnectorWorkerMetadata":
        assert isinstance(other, AttnWorkerMeta)
        self.emitted |= other.emitted
        return self


def _ranges_of(new_req) -> list | None:
    """Per-request prompt ranges, or None.

    Passed by the caller as::

        SamplingParams(extra_args={"kv_transfer_params": {"ranges": [[lo, hi], ...]}})

    which is vLLM's standard per-request channel to a KV connector. Anything
    malformed is ignored rather than raised: provenance capture must never be
    able to fail a generation request.
    """
    sp = getattr(new_req, "sampling_params", None)
    extra = getattr(sp, "extra_args", None) or {}
    params = extra.get("kv_transfer_params") or {}
    raw = params.get("ranges") if isinstance(params, dict) else None
    if not raw:
        return None
    try:
        out = [(int(a), int(b)) for a, b in raw if int(b) > int(a)]
    except (TypeError, ValueError):
        logger.warning("attn_connector: ignoring malformed ranges %r", raw)
        return None
    return out or None


class _SchedulerSide:
    def __init__(self) -> None:
        self._finished: list[str] = []
        self._seen: set[str] = set()

    def request_finished(self, request: "Request") -> tuple[bool, dict[str, Any] | None]:
        if request.request_id in self._seen:
            self._finished.append(request.request_id)
        return False, None

    def build_connector_meta(self, scheduler_output: "SchedulerOutput") -> AttnMetadata:
        meta, self._finished = AttnMetadata(finished=self._finished), []
        counts = scheduler_output.num_scheduled_tokens
        cached = scheduler_output.scheduled_cached_reqs
        # (req_id, blocks, num_computed, prompt_token_ids, ranges). Ranges ride
        # in on the new-request path only: they describe the prompt, which does
        # not change, and the worker freezes the plan on the first decode step.
        incoming = [(r.req_id, r.block_ids, r.num_computed_tokens, r.prompt_token_ids,
                     _ranges_of(r))
                    for r in scheduler_output.scheduled_new_reqs]
        incoming += [(rid, cached.new_block_ids[i], cached.num_computed_tokens[i],
                      None, None)
                     for i, rid in enumerate(cached.req_ids)]
        for req_id, blocks, computed, prompt_ids, ranges in incoming:
            n = counts.get(req_id, 0)
            if not n:
                continue
            self._seen.add(req_id)
            meta.scheduled.append(StepRequest(
                request_id=req_id,
                new_block_ids=tuple(list(g) for g in blocks) if blocks else (),
                num_computed_tokens=computed, num_scheduled_tokens=n,
                prompt_token_ids=list(prompt_ids) if prompt_ids else None,
                ranges=ranges))
        return meta

    def update_connector_output(self, connector_output: "KVConnectorOutput") -> None:
        meta = connector_output.kv_connector_worker_meta
        if isinstance(meta, AttnWorkerMeta):
            self._seen -= meta.emitted

    def has_pending_push_work(self) -> bool:
        return bool(self._finished)


@dataclass
class _Group:
    gid: int
    layers: list[LayerLayout]
    names: list[str]
    kind: str
    window: int | None


class _RequestState:
    """Per-request attribution, alive only while the request is in flight.

    In top-k mode the only per-step storage is k (position, score) pairs, so the
    footprint is O(steps * k) rather than O(steps * T). At T=95k,
    G=5k that is ~3 MB instead of ~7.6 GB, which is the difference between
    working and an immediate OOM.
    """

    __slots__ = ("blocks", "table", "tables", "scratch", "colsum", "pos", "val",
                 "resid",
                 "prompt_len", "steps", "dropped", "prompt_token_ids", "event", "started",
                 "seen_computed", "restarts", "k", "plan", "seg_idx", "seg_live",
                 "seg_want", "seg_kmax", "ranges", "n_keys", "owner", "head",
                 "lsum", "lwins", "hent", "hdev", "hcnt", "hpos", "hval", "hmass")

    def __init__(self, prompt_token_ids: list[int] | None):
        # Block ids are PER KV CACHE GROUP. A uniform model has one group and
        # the distinction is invisible, but a hybrid one -- full attention
        # interleaved with gated-delta-net layers, as in Qwen3.5 -- has
        # several, and the full-attention group is not necessarily index 0.
        # Taking group 0's ids for every group indexes the attention cache
        # with another group's block numbers: the kernel then reads unrelated
        # memory and the scores come back NaN, with nothing logged.
        self.blocks: dict[int, list[int]] = {}
        self.tables: dict[int, torch.Tensor] = {}
        self.table: torch.Tensor | None = None     # group 0, kept for callers
        self.scratch: torch.Tensor | None = None   # (groups, 2, T) acc-mean, running max
        self.colsum: torch.Tensor | None = None    # (groups, 2, T) running sum / max
        self.pos: torch.Tensor | None = None       # (groups, steps, k) int32
        self.val: torch.Tensor | None = None       # (groups, N_AGG, steps, k)
        self.resid: torch.Tensor | None = None     # (groups, steps) unretained mean mass
        self.prompt_len = 0
        self.steps = 0
        self.dropped = 0
        self.prompt_token_ids = prompt_token_ids or []
        self.event: torch.cuda.Event | None = None
        self.started = time()
        self.seen_computed = -1
        self.restarts = 0
        self.k: int | None = None   # entries stored per step, frozen on first decode step
        self.plan: list[tuple[int, int, int]] = []   # [(lo, hi, keep)]
        self.seg_idx = None
        self.seg_live = None
        self.seg_want = None
        self.seg_kmax = 0
        self.ranges: list | None = None   # caller-declared, if any
        self.n_keys = 0
        self.owner = None
        self.head = None
        self.lsum = None
        self.lwins = None
        self.hent = None
        self.hdev = None
        self.hcnt = None
        self.hpos = None
        self.hval = None
        self.hmass = None

    def step_capacity(self, max_steps: int) -> int:
        """Rows to have allocated before recording step ``self.steps``.

        Bounded mode allocates the cap once. Unbounded mode doubles from
        STEP_CHUNK, so memory tracks the generation actually produced rather
        than a worst case nobody hits.
        """
        if max_steps:
            return max_steps
        need = self.steps + 1
        cap = STEP_CHUNK
        while cap < need:
            cap *= 2
        return cap

    def _grow_steps(self, cap: int) -> None:
        """Extend the step dimension of every step-indexed buffer to ``cap``."""
        def ext(t, dim):
            if t is None or t.shape[dim] >= cap:
                return t
            shape = list(t.shape)
            shape[dim] = cap
            out = torch.full(shape, -1, dtype=t.dtype, device=t.device) \
                if t.dtype == torch.int32 else torch.zeros(shape, dtype=t.dtype,
                                                           device=t.device)
            idx = [slice(None)] * t.dim()
            idx[dim] = slice(0, t.shape[dim])
            out[tuple(idx)] = t
            return out
        self.pos = ext(self.pos, 1)
        self.head = ext(self.head, 1)
        self.val = ext(self.val, 2)
        self.resid = ext(self.resid, 1)
        self.hpos = ext(self.hpos, 2)
        self.hval = ext(self.hval, 2)

    def alloc(self, groups: int, max_steps: int, k: int, n_keys: int, device,
              layer_stats: bool = False, n_layers: int = 0, n_heads: int = 0,
              tp_size: int = 1) -> None:
        steps_cap = self.step_capacity(max_steps)
        if self.scratch is None or self.scratch.shape[2] < n_keys:
            cap = max(256, n_keys)
            self.scratch = torch.zeros((groups, 2, cap), dtype=torch.float32, device=device)
            grown = torch.zeros((groups, 2, cap), dtype=torch.float32, device=device)
            if self.colsum is not None:
                grown[:, :, : self.colsum.shape[2]] = self.colsum
            self.colsum = grown
        # Width must track k: after a preemption restart the plan is recomputed
        # and may differ, so a merely-zeroed buffer of the old width would not fit.
        # Width must track n_keys: a restart refreezes the plan against a
        # possibly different prompt length.
        if layer_stats and self.lwins is None:
            self.lsum = torch.zeros((groups, n_layers), dtype=torch.float64,
                                    device=device)
            # [bucket, groups, layer*heads]: bucket 0 = the first decode token,
            # bucket 1 = every later one. Splitting them tests whether a
            # selection made on step 0 would transfer to the rest.
            # Indexed by the owner code, which under TP is a *global* head id,
            # so this must span every rank's heads or the bincount overruns it.
            self.lwins = torch.zeros((2, groups, n_layers * n_heads * max(1, tp_size)),
                                     dtype=torch.float64, device=device)
            self.hent = torch.zeros((groups, n_layers * n_heads),
                                    dtype=torch.float64, device=device)
            self.hdev = torch.zeros((groups, n_layers * n_heads),
                                    dtype=torch.float64, device=device)
            self.hcnt = torch.zeros(groups, dtype=torch.float64, device=device)
            self.hpos = torch.full((groups, n_layers * n_heads, steps_cap), -1,
                                   dtype=torch.int32, device=device)
            self.hval = torch.zeros((groups, n_layers * n_heads, steps_cap),
                                    dtype=torch.float32, device=device)
            self.hmass = torch.zeros((groups, n_layers * n_heads),
                                     dtype=torch.float64, device=device)
        if self.owner is not None and self.owner.shape[1] < n_keys:
            self.owner = None
        if self.owner is None:
            self.owner = torch.full((groups, max(256, n_keys)), -1,
                                    dtype=torch.int32, device=device)
        if k and self.pos is not None and self.pos.shape[2] != k:
            self.pos = self.val = self.resid = self.head = None
        if k and self.pos is None:
            self.pos = torch.zeros((groups, steps_cap, k), dtype=torch.int32,
                                   device=device)
            self.head = torch.full((groups, steps_cap, k), -1, dtype=torch.int32,
                                   device=device)
            self.val = torch.zeros((groups, N_AGG, steps_cap, k),
                                   dtype=torch.float32, device=device)
            self.resid = torch.zeros((groups, steps_cap), dtype=torch.float32,
                                     device=device)
        # Unbounded mode: extend in place once the generation outruns the
        # current allocation.
        self._grow_steps(steps_cap)

    def reset_buffers(self) -> None:
        # Drop, not zero: a restart recomputes the selection plan, so the
        # per-step buffers may need a different width.
        self.pos = self.val = self.resid = None
        self.owner = self.head = None
        self.hpos = self.hval = None
        self.plan = []
        self.seg_idx = self.seg_live = self.seg_want = None
        self.seg_kmax = 0
        for t in (self.lsum, self.lwins, self.hent, self.hdev, self.hcnt, self.hmass):
            if t is not None:
                t.zero_()
        if self.colsum is not None:
            self.colsum.zero_()

    def note_progress(self, num_computed: int) -> bool:
        """Detect a preemption restart. Returns True if state was reset.

        On preemption vLLM frees the request's blocks and recomputes prefill on
        resume, handing back a *fresh* block list. Appending that to the stale
        one leaves the block table addressing recycled memory, which reads as
        garbage. `num_computed_tokens` going backwards is the signal.
        """
        if num_computed < self.seen_computed:
            self.blocks.clear()
            self.table = None
            self.prompt_len = 0
            self.steps = 0
            self.restarts += 1
            self.k = None
            self.n_keys = 0
            # `ranges` deliberately survives: it describes the prompt, which a
            # preemption does not change. The plan derived from it is rebuilt.
            self.reset_buffers()
            self.seen_computed = num_computed
            return True
        self.seen_computed = max(self.seen_computed, num_computed)
        return False


class _WorkerSide:
    def __init__(self, kv_cache_config, conf, tp_size, workflow_id, parent_workflow_id,
                 max_steps: int, top_pct: float, chunk: int,
                 out_dir: str = DEFAULT_OUT_DIR, checksum: bool = True,
                 layer_stats: bool = False) -> None:
        from flowcept.flowceptor.adapters.vllm.vllm_interceptor import VLLMInterceptor

        self._interceptor = VLLMInterceptor.get_instance()
        self._interceptor.start(bundle_exec_id=workflow_id)
        self._workflow_id = workflow_id
        self._parent_workflow_id = parent_workflow_id
        self._conf = conf
        self._tp_size = tp_size
        self._tp_rank = 0
        self._max_steps = max_steps
        self._top_pct = max(0.0, float(top_pct))
        self._chunk = max(0, int(chunk))
        self._out_dir = Path(out_dir).expanduser().resolve()
        self._checksum = bool(checksum)
        self._layer_stats = bool(layer_stats)
        self._missing_q = 0

        self._groups = [(list(g.layer_names), g.kv_cache_spec)
                        for g in kv_cache_config.kv_cache_groups]
        self._num_groups = len(self._groups)
        self._num_layers_total = sum(len(n) for n, _ in self._groups)
        self._layouts: dict[str, LayerLayout] = {}
        self._scored: list[_Group] = []
        self._num_q_heads = 0
        self._dumped = False
        self._warned_table = False
        self._block_size = 0
        self._state: dict[str, _RequestState] = {}
        self._emitted: set[str] = set()
        self._stream: torch.cuda.Stream | None = None
        self._scratch: torch.Tensor | None = None
        self._enabled = False

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        for layer_names, spec in self._groups:
            nkv = getattr(spec, "num_kv_heads", None)
            if nkv is None:
                continue
            for name in layer_names:
                cache = kv_caches.get(name)
                if cache is None:
                    continue
                try:
                    self._layouts[name] = resolve_layer_layout(
                        name, cache, block_size=spec.block_size, num_kv_heads=nkv,
                        head_size=spec.head_size,
                        head_size_v=getattr(spec, "head_size_v", None),
                    )
                except UnsupportedLayout as exc:
                    logger.warning("attn_connector: %s", exc)

        for gid, (layer_names, spec) in enumerate(self._groups):
            kept = [n for n in layer_names if n in self._layouts]
            if kept:
                self._scored.append(_Group(
                    gid=gid, layers=[self._layouts[n] for n in kept], names=kept,
                    **dict(zip(("kind", "window"), _attn_spec(spec))),
                ))

        if self._tp_size > 1:
            from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

            self._tp_rank = get_tensor_model_parallel_rank()

        if not self._scored:
            logger.warning("attn_connector: no supported attention layers; capture disabled")
            return
        if not REGISTRY.enabled:
            logger.error(
                "attn_connector: the query probe is not installed, so no decode "
                "queries will arrive and nothing can be scored. Call "
                "vllm_attn_connector.install_probe() BEFORE constructing the engine."
            )
            return

        self._num_q_heads = self._infer_query_heads()
        for g in self._scored:
            if g.kind != FULL_ATTENTION:
                logger.warning(
                    "attn_connector: group %d is %s; beyond the window these layers "
                    "cannot attend to a prompt token at all. Recorded separately.",
                    g.gid, g.kind,
                )
        self._block_size = self._scored[0].layers[0].block_size
        self._stream = torch.cuda.Stream()
        self._enabled = True
        if self._tp_rank == 0:
            self._interceptor.send_model_workflow(
                self._workflow_id, self._conf,
                parent_workflow_id=self._parent_workflow_id,
                attention_config=self._attention_config()
            )
        logger.info(
            "attn_connector: %d layers in %d group(s), %d query heads / %d kv heads, "
            "max_steps=%d, selection=%s, tp_rank=%d/%d",
            sum(len(g.layers) for g in self._scored), len(self._scored),
            self._num_q_heads, self._scored[0].layers[0].num_kv_heads,
            self._max_steps,
            (f'top {self._top_pct}% per {self._chunk}-token segment'
             if self._chunk else f'top {self._top_pct}% per prompt'),
            self._tp_rank, self._tp_size,
        )

    def _infer_query_heads(self) -> int:
        """Query heads per rank, or a loud complaint.

        Three sources, in descending reliability. The last one -- assuming the
        model is not using GQA at all -- is a guess, and guessing silently is
        what let Qwen3.5 run with 2 query heads instead of 16: every scored
        value was then computed over one eighth of the heads, with nothing in
        the log to say so.
        """
        n = self._conf.get("num_attention_heads")
        if n:
            return int(n) // max(1, self._tp_size)
        try:
            from transformers import AutoConfig

            cfg = AutoConfig.from_pretrained(
                self._conf["model"], trust_remote_code=self._conf.get("trust_remote_code", False)
            )
            # Multimodal configs nest the language model's shape.
            cfg = getattr(cfg, "text_config", None) or cfg
            return int(cfg.num_attention_heads) // max(1, self._tp_size)
        except Exception as exc:                            # noqa: BLE001
            fallback = self._scored[0].layers[0].num_kv_heads
            logger.warning(
                "attn_connector: could not determine num_attention_heads for "
                "%s (%s); assuming %d, i.e. no GQA. If the model does use GQA "
                "this is WRONG and only %d of its query heads will be scored.",
                self._conf.get("model"), exc, fallback, fallback)
            return fallback

    # -- per-step ----------------------------------------------------------- #

    def _warn_no_table(self, gid: int, tables: dict) -> None:
        """Complain once. Scoring a group with no block table of its own would
        mean reading the cache with another group's block numbers."""
        if not getattr(self, "_warned_table", False):
            self._warned_table = True
            logger.warning(
                "attn_connector: no block table for kv cache group %d "
                "(have groups %s). This request will not be scored.",
                gid, sorted(tables))

    def _warn_no_query(self, layer_name: str, rows: int) -> None:
        """Complain once. A silent skip here produces all-zero records that pass
        every structural check -- mass still "conserves", because 0 + 1 == 1."""
        self._missing_q += 1
        if self._missing_q == 1:
            logger.warning(
                "attn_connector: no decode queries for layer %s at %d row(s) "
                "(buffer capacity %d). Scores for this step will be zero. If "
                "this persists the probe is not reaching the forward -- check "
                "install_probe() ran before the engine was built.",
                layer_name, rows, REGISTRY.max_rows)

    def score_step(self, meta: AttnMetadata) -> None:
        """Score the decode queries this step produced, on the side stream."""
        if not self._enabled or not meta.scheduled:
            return
        device = self._scored[0].layers[0].k_view.device
        stream = self._stream
        assert stream is not None
        stream.wait_stream(torch.cuda.current_stream())

        # vLLM v1 orders decode requests first, one token each, so row i of the
        # captured queries is the i-th decoding request in scheduler order.
        #
        # num_scheduled_tokens == 1 is necessary but NOT sufficient: chunked
        # prefill can leave a 1-token remainder, which looks identical from here
        # but arrives while the prompt is still incomplete. Require that the
        # prompt is fully computed as well.
        def _is_decode(sr: StepRequest) -> bool:
            if sr.num_scheduled_tokens != 1:
                return False
            st = self._state.get(sr.request_id)
            n_prompt = len(st.prompt_token_ids) if st else 0
            return sr.num_computed_tokens >= n_prompt if n_prompt else True

        decoding = [s for s in meta.scheduled if _is_decode(s)]
        # Under CUDA graphs the probe's Python does not run on replay, so it has
        # no count to cross-check against; the scheduler's is the only one. When
        # it *did* run (eager), disagreement means the batch is not laid out the
        # way decode assumes, so drop the step rather than mis-attribute.
        if decoding and REGISTRY.ran_eagerly:
            n_probe = REGISTRY.num_decode_reqs
            if n_probe != len(decoding):
                logger.warning(
                    "attn_connector: probe saw %d decode rows but the scheduler lists %d; "
                    "skipping this step rather than mis-attributing.", n_probe, len(decoding)
                )
                decoding = []
        n_rows = len(decoding)

        with torch.cuda.stream(stream):
            for step_req in meta.scheduled:
                st = self._state.get(step_req.request_id)
                if st is None:
                    st = _RequestState(step_req.prompt_token_ids)
                    st.ranges = step_req.ranges
                    self._state[step_req.request_id] = st
                elif step_req.prompt_token_ids:
                    st.prompt_token_ids = step_req.prompt_token_ids
                if st.note_progress(step_req.num_computed_tokens):
                    logger.warning(
                        "attn_connector: %s restarted (preemption); discarding "
                        "%d recorded step(s) and re-tracking from scratch.",
                        step_req.request_id, st.restarts,
                    )
                if step_req.new_block_ids:
                    for _gi, _new in enumerate(step_req.new_block_ids):
                        acc = st.blocks.setdefault(_gi, [])
                        acc.extend(_new)
                        st.tables[_gi] = torch.tensor(acc, dtype=torch.int32,
                                                      device=device)
                    st.table = st.tables.get(0)
                if step_req.num_scheduled_tokens > 1 or st.prompt_len == 0:
                    # Still prefilling: freeze the prompt length as it grows.
                    st.prompt_len = max(
                        st.prompt_len,
                        step_req.num_computed_tokens + step_req.num_scheduled_tokens,
                    )

            for row_i, step_req in enumerate(decoding):
                st = self._state[step_req.request_id]
                if not st.tables or st.prompt_len == 0:
                    continue
                if self._max_steps and st.steps >= self._max_steps:
                    st.dropped += 1
                    continue
                # Blocks OF THE GROUP BEING SCORED. `st.blocks` is keyed by
                # kv cache group, so its len() is the number of groups, not
                # the number of blocks -- reading it directly capped every
                # prompt at groups * block_size (4 * 1056 = 4224 on Qwen3.5),
                # silently truncating the longest prompts and dropping the
                # declared ranges past the cut.
                _sg = self._scored[0].gid
                n_keys = min(st.prompt_len,
                             len(st.blocks.get(_sg, ())) * self._block_size)
                if st.n_keys:
                    # Frozen with k on the first decode step: the selection plan
                    # and the buffers are sized to it.
                    if n_keys != st.n_keys:
                        logger.warning(
                            "attn_connector: %s prompt length moved %d -> %d after "
                            "scoring began; clamping.", step_req.request_id,
                            st.n_keys, n_keys)
                    n_keys = st.n_keys
                if st.k is None:
                    # Frozen for the request: n_keys settles when prefill ends
                    # and the buffers are sized to it. `ranges`, when the caller
                    # declared any, makes the segments follow the prompt's own
                    # structure instead of a uniform grid.
                    st.plan = _segment_plan(n_keys, self._chunk, self._top_pct,
                                            st.ranges)
                    (st.seg_idx, st.seg_live,
                     st.seg_want, st.k) = _segment_index(st.plan, device)
                    st.seg_kmax = max(k for _, _, k in st.plan)
                    st.n_keys = n_keys
                k = st.k
                st.alloc(len(self._scored), self._max_steps, k, n_keys, device,
                         self._layer_stats, max(len(g.layers) for g in self._scored),
                         self._num_q_heads, self._tp_size)
                g_idx = st.steps
                for gi, group in enumerate(self._scored):
                    row = st.scratch[gi, :, :n_keys]
                    row.zero_()
                    st.owner[gi, :n_keys].fill_(-1)
                    for li, layout in enumerate(group.layers):
                        q = REGISTRY.get(layout.layer_name, n_rows)
                        if q is None:
                            self._warn_no_query(layout.layer_name, n_rows)
                            continue
                        table = st.tables.get(group.gid)
                        if table is None:
                            self._warn_no_table(group.gid, st.tables)
                            continue
                        probs = decode_attention(
                            q[row_i], layout.k_view, table, n_keys,
                            num_query_heads=self._num_q_heads,
                        )
                        # OPAL_ATTN_DEBUG=1 dumps the first scored row. An
                        # all-zero capture is otherwise indistinguishable from
                        # a working one until the files are read back, and the
                        # shapes here say immediately whether the problem is
                        # upstream (q or the cache view) or downstream
                        # (selection).
                        if _DEBUG and not self._dumped:
                            self._dumped = True
                            logger.warning(
                                "attn_connector DEBUG layer=%s gid=%d "
                                "groups_with_blocks=%s table[:4]=%s "
                                "table.max=%s n_blocks_in_cache=%d "
                                "q_row=%s k_view=%s n_keys=%d num_q_heads=%d "
                                "probs=%s probs.max=%.6g row0.sum=%.6g "
                                "qrow.absmax=%.6g",
                                layout.layer_name, group.gid, sorted(st.tables),
                                table[:4].tolist(), int(table.max()),
                                layout.k_view.shape[0],
                                tuple(q[row_i].shape),
                                tuple(layout.k_view.shape), n_keys,
                                self._num_q_heads, tuple(probs.shape),
                                float(probs.max()), float(probs[0].sum()),
                                float(q[row_i].abs().max()))
                        if self._layer_stats and n_keys > 1:
                            _ = None
                            # Per-head shape statistics, sink excluded: it takes
                            # ~40% of every head's mass and says nothing about
                            # which head is discriminative.
                            pn_raw = probs[:, 1:]
                            pn = pn_raw / pn_raw.sum(1, keepdim=True).clamp_min(1e-12)
                            base = li * probs.shape[0]
                            st.hent[gi, base:base + probs.shape[0]] += (
                                -(pn * torch.log(pn.clamp_min(1e-12))).sum(1)).double()
                            st.hdev[gi, base:base + probs.shape[0]] += (
                                (pn - pn.mean(0, keepdim=True)).abs().sum(1)).double()
                            # Where each head looks, per decode token -- computed
                            # over NON-SINK positions, consistently with the
                            # entropy above. Including position 0 makes this
                            # measure sink behaviour: 66% of all head argmaxes
                            # land there, and 99.8% for the peakiest heads.
                            hv, hp = pn_raw.max(1)
                            st.hpos[gi, base:base + probs.shape[0], g_idx] = (
                                hp + 1).to(torch.int32)
                            st.hval[gi, base:base + probs.shape[0], g_idx] = hv.float()
                            # How much of the head's attention is off the sink at
                            # all -- a head with none of it cannot be attributing.
                            st.hmass[gi, base:base + probs.shape[0]] += (
                                pn_raw.sum(1)).double()
                            if li == 0:
                                # once per step, not per layer: each head slot
                                # accumulates one value per decode token
                                st.hcnt[gi] += 1
                            # Which (layer, head) supplied the running max, per
                            # prefill position. The emitted max is reduced over
                            # both, so the winner's identity is otherwise lost.
                            lmax, hidx = probs.max(0)
                            st.lsum[gi, li] += float(lmax.sum())
                        else:
                            lmax, hidx = probs.max(0)
                        if self._layer_stats and n_keys <= 1:
                            lmax, hidx = probs.max(0)
                        # Which (layer, head) currently owns the max at each
                        # prefill position. Encoded as layer*H + head.
                        better = lmax > row[1]
                        st.owner[gi, :n_keys] = torch.where(
                            better, (li * probs.shape[0] + hidx).to(torch.int32),
                            st.owner[gi, :n_keys])
                        row[0] += probs.mean(0)
                        torch.maximum(row[1], lmax, out=row[1])
                    row[0] /= max(1, len(group.layers))
                    if self._tp_size > 1:
                        # Each rank owns a head slice, so the row must be whole
                        # before selection reads it -- and the owner must be
                        # re-encoded into a global head index, or topk_head
                        # would name a head on whichever rank happened to emit.
                        red, own = _reduce_row(row, st.owner[gi, :n_keys],
                                               self._num_q_heads)
                        row.copy_(red)
                        st.owner[gi, :n_keys] = own
                    # Column aggregates are O(T) once, not per step: keep them
                    # in both modes so the summed view is never lost.
                    st.colsum[gi, 0, :n_keys] += row[0]
                    torch.maximum(st.colsum[gi, 1, :n_keys], row[1],
                                  out=st.colsum[gi, 1, :n_keys])
                    if st.lwins is not None:
                        w = st.owner[gi, :n_keys]
                        b = 0 if g_idx == 0 else 1
                        st.lwins[b, gi] += torch.bincount(
                            w[w >= 0].long(), minlength=st.lwins.shape[2]).double()
                    if k:
                        # Select on the MAX aggregation: a mean over every
                        # (layer, head) pair buries whichever head is doing the
                        # work, which is the one worth keeping.
                        idx, val = _segmented_topk(row[1], st.seg_idx, st.seg_live,
                                                   st.seg_want, k, st.seg_kmax)
                        keep = idx >= 0
                        st.pos[gi, g_idx, :k] = idx.to(torch.int32)
                        gather = idx.clamp_min(0)
                        st.val[gi, 0, g_idx, :k] = val                      # all_max
                        mean_v = torch.where(keep, row[0][gather], 0.0)
                        st.val[gi, 1, g_idx, :k] = mean_v                   # all_avg
                        # The (layer, head) that supplied the max at each
                        # retained position. Free: the max reduction already
                        # tracked the winner in st.owner.
                        st.head[gi, g_idx, :k] = torch.where(
                            keep, st.owner[gi, :n_keys][gather],
                            torch.full_like(idx, -1, dtype=torch.int32)).to(torch.int32)

                        # The mean row is a distribution, so whatever selection
                        # did not retain stays recoverable as one scalar.
                        st.resid[gi, g_idx] = 1.0 - mean_v.sum()
                st.steps += 1

            event = torch.cuda.Event()
            event.record(stream)
        for step_req in meta.scheduled:
            st = self._state.get(step_req.request_id)
            if st is not None:
                st.event = event

    # -- emission ----------------------------------------------------------- #

    def emit_finished(self, meta: AttnMetadata) -> None:
        for req_id in meta.finished:
            st = self._state.pop(req_id, None)
            self._emitted.add(req_id)
            if st is None or st.colsum is None or st.steps == 0:
                continue
            try:
                if self._tp_rank == 0 or self._tp_size > 1:
                    self._emit(req_id, st)
            except Exception:
                logger.exception("attn_connector: failed to emit %s", req_id)

    def _emit(self, req_id: str, st: _RequestState) -> None:
        if st.event is not None:
            st.event.synchronize()
        n_keys = min(st.prompt_len, st.colsum.shape[2])
        if self._tp_rank != 0:
            return
        G, k = st.steps, (st.pos.shape[2] if st.pos is not None else 0)
        npy = lambda t, dt: t.detach().cpu().numpy().astype(dt, copy=False)

        for gi, group in enumerate(self._scored):
            tensors = {
                "attn_sum": npy(st.colsum[gi, 0, :n_keys], "float32"),
                "attn_peak": npy(st.colsum[gi, 1, :n_keys], "float32"),
                "prompt_token_ids": np.asarray(st.prompt_token_ids, dtype="int32"),
            }
            n_bad = 0
            if k:
                mean = st.val[gi, 1, :G]
                # Non-finite rows can only come from scoring recycled blocks;
                # drop them rather than let one poisoned step propagate.
                finite = torch.isfinite(mean).all(dim=1)
                n_bad = int((~finite).sum())
                if n_bad:
                    logger.warning("attn_connector: %s group %d: %d/%d step(s) "
                                   "non-finite, zeroed.", req_id, group.gid, n_bad, G)
                    mean = torch.where(finite[:, None], mean, 0.0)
                tensors["topk_pos"] = npy(st.pos[gi, :G], "int32")
                tensors["topk_head"] = npy(st.head[gi, :G], "int32")
                tensors["val_all_max"] = npy(st.val[gi, 0, :G], "float32")
                tensors["val_all_avg"] = npy(mean, "float32")
                # [G, 1] rather than flat: it is per decode token, not per
                # prefill token, and a flat vector would be read as the latter.
                tensors["topk_residual"] = npy(st.resid[gi, :G, None], "float32")
                # The partition every retained position belongs to, (lo, hi,
                # keep), whether it came from a uniform grid or the caller's
                # declared ranges.
                tensors["segments"] = np.asarray(st.plan, dtype="int32").reshape(-1, 3)
            if self._layer_stats and st.lwins is not None:
                tensors["head_argmax_pos"] = npy(st.hpos[gi, :, :G], "int32")
                tensors["head_argmax_val"] = npy(st.hval[gi, :, :G], "float32")
                cnt = st.hcnt[gi].clamp_min(1)
                tensors["head_offsink_mass"] = npy(st.hmass[gi] / cnt, "float32")
                tensors["head_entropy"] = npy(st.hent[gi] / cnt, "float32")
                tensors["head_deviation"] = npy(st.hdev[gi] / cnt, "float32")
                tensors["wins_first_step"] = npy(st.lwins[0, gi], "float32")
                tensors["wins_later_steps"] = npy(st.lwins[1, gi], "float32")
                tensors["layer_maxsum"] = npy(st.lsum[gi], "float32")

            mode = "variable" if st.ranges else "fixed"
            # The header makes the file interpretable on its own, away from the
            # provenance store that references it.
            desc = store.write(
                self._out_dir, self._workflow_id, req_id, group.gid, tensors,
                header={"request_id": req_id, "workflow_id": self._workflow_id,
                        "kv_cache_group_id": str(group.gid), "segment_mode": mode,
                        "top_pct": str(self._top_pct),
                        "no_entry_sentinel": str(NO_ENTRY),
                        "head_code": self._head_code()},
                checksum=self._checksum)
            desc.update(kv_cache_group_id=group.gid,
                        written_by_tp_rank=self._tp_rank,
                        segment_mode=mode,
                        decode_steps_dropped=st.dropped,
                        decode_steps_nonfinite=n_bad,
                        restarts=st.restarts)

            self._interceptor.capture_request(
                workflow_id=self._workflow_id,
                request_id=f"{req_id}:g{group.gid}",
                attention_stats=desc,
                num_prompt_tokens=n_keys,
                num_decode_tokens=G,
                started_at=st.started,
                activity=ACTIVITY,
            )

    def _head_code(self) -> str:
        """How to read `topk_head`: a global head index in both TP regimes.

        Under TP the owner is reduced alongside the value (see `_reduce_row`),
        so the id always names the head that actually produced `val_all_max`,
        whichever rank held it. `num_query_heads` in this config is per-rank, so
        the multiplier here is the global count.
        """
        return f"layer * {self._num_q_heads * self._tp_size} + head"

    def _attention_config(self) -> dict[str, Any]:
        """Settings constant for this engine, recorded once on the workflow.

        Per-request values stay on the task: `segment_mode` depends on whether
        that request declared ranges, and the health counters are how a single
        bad record is identified.
        """
        return {
            "connector": "vllm-attn-connector",
            "metric": "decode_attention",
            "metric_reference": "exact softmax(q_t . K_prompt^T / sqrt(d))",
            "selected_on": "val_all_max",
            "top_pct": self._top_pct,
            "chunk_size": self._chunk,
            "max_steps": self._max_steps,
            "no_entry_sentinel": NO_ENTRY,
            "head_code": self._head_code(),
            "num_layers_scored": max((len(g.layers) for g in self._scored), default=0),
            # Needed to decode topk_head, so it is a field rather than only a
            # number embedded in head_code. Per-rank under TP, matching how the
            # ids are encoded.
            "num_query_heads": self._num_q_heads,
            "attention_window": self._scored[0].window if self._scored else None,
            "sink_excluded_from_selection": True,
            "tensor_parallel_size": self._tp_size,
            "layer_stats": self._layer_stats,
            "out_dir": str(self._out_dir),
        }

    def take_emitted(self) -> AttnWorkerMeta | None:
        if not self._emitted:
            return None
        meta = AttnWorkerMeta(emitted=self._emitted)
        self._emitted = set()
        return meta

    def shutdown(self) -> None:
        self._interceptor.stop()


def _segment_plan(n_keys: int, chunk: int, pct: float,
                  ranges: list | None = None) -> list[tuple[int, int, int]]:
    """Partition [0, n_keys) into segments and decide how many to keep in each.

    Returns ``[(lo, hi, keep), ...]`` covering the prompt with no gaps and no
    overlaps. Two ways to segment, same output shape, so everything downstream
    is identical:

    *fixed* -- uniform blocks of ``chunk`` tokens. Used when the caller has no
    structure to declare.

    *variable* -- the caller's own ``ranges``, e.g. one per retrieved document
    or tool output. Ranges are clamped to the prompt, sorted, and the gaps
    between them become segments of their own so the partition stays complete.
    Overlapping ranges are merged by truncation: a range starting inside the
    previous one resumes at its end.

    Either way ``keep`` is ``pct`` percent of the segment, rounded, floored at
    1 -- so every segment is represented no matter how short. That is the point
    of segmenting at all: a whole-prompt top-X% concentrates wherever the
    distribution peaks and can leave entire regions with no stored entry,
    which makes those regions invisible downstream.
    """
    segs: list[tuple[int, int]] = []
    if ranges:
        cur = 0
        for lo, hi in sorted((max(0, int(a)), min(n_keys, int(b))) for a, b in ranges):
            lo = max(lo, cur)
            if hi <= lo:
                continue
            if lo > cur:
                segs.append((cur, lo))
            segs.append((lo, hi))
            cur = hi
        if cur < n_keys:
            segs.append((cur, n_keys))
    else:
        c = min(chunk, n_keys) if chunk else n_keys
        c = max(1, c)
        segs = [(i, min(i + c, n_keys)) for i in range(0, n_keys, c)]
    return [(lo, hi, max(1, round(pct / 100.0 * (hi - lo)))) for lo, hi in segs]


def _segment_index(plan: list[tuple[int, int, int]], device):
    """Precompute the gather for `_segmented_topk`, once per request.

    Returns (idx, live, want, total): a [n_seg, wmax] index matrix, a validity
    mask for the ragged tail of each row, the per-segment quota, and the total
    number of entries a step will produce.
    """
    wmax = max(hi - lo for lo, hi, _ in plan)
    n = len(plan)
    idx = torch.zeros((n, wmax), dtype=torch.long, device=device)
    live = torch.zeros((n, wmax), dtype=torch.bool, device=device)
    for i, (lo, hi, _) in enumerate(plan):
        w = hi - lo
        idx[i, :w] = torch.arange(lo, hi, device=device)
        live[i, :w] = True
    want = torch.tensor([[k] for _, _, k in plan], device=device)
    return idx, live, want, sum(k for _, _, k in plan)


def _segmented_topk(row: torch.Tensor, idx, live, want, total: int, kmax: int):
    """Top-`keep` positions inside each segment, flat and ordered by position.

    Unused slots carry position NO_ENTRY and value 0 so the output stays a
    fixed-width rectangle.

    **Position 0 is never selected.** It is the attention sink: in decoder-only
    LLMs the first token absorbs a large, roughly content-independent share of
    attention (arXiv:2309.17453), so it wins slots without saying anything about
    what the model is doing. Its mass stays recoverable from `attn_sum` and
    `attn_peak`, which cover every position.
    """
    neg = torch.finfo(row.dtype).min
    pad = torch.where(live, row[idx], neg)
    if idx[0, 0] == 0:
        pad[0, 0] = neg
    val, loc = torch.topk(pad, kmax, dim=1)
    pos = torch.gather(idx, 1, loc)
    keep = (torch.arange(kmax, device=row.device).unsqueeze(0) < want) & (val > neg)
    pos = torch.where(keep, pos, torch.full_like(pos, NO_ENTRY))
    val = torch.where(keep, val, torch.zeros_like(val))
    pos, val = pos.reshape(-1), val.reshape(-1)
    # Order by position: makes range aggregation a searchsorted, not a scan.
    order = torch.argsort(torch.where(pos < 0, torch.full_like(pos, 1 << 30), pos))
    return pos[order][:total].to(torch.int32), val[order][:total]


def _resolve_owner(V: torch.Tensor, O: torch.Tensor, heads_per_rank: int):
    """Pick the winning rank per position and re-encode its head as global.

    `V` and `O` are [world, n_keys]: each rank's max and the local code of the
    head that produced it. A local code is ``layer * heads_per_rank + head``,
    which means nothing outside its rank, so the winner's is rebuilt as::

        layer * (heads_per_rank * world) + rank * heads_per_rank + head

    Split out of `_reduce_row` because it is the part that can silently be
    wrong, and because it is testable: the collectives need a process group and
    more than one GPU, this index arithmetic needs neither. The `-1` used before
    any head has claimed a position passes through untouched.
    """
    best = V.argmax(0)
    win = O.gather(0, best.unsqueeze(0)).squeeze(0)
    glob = ((win // heads_per_rank) * (heads_per_rank * V.shape[0])
            + best * heads_per_rank + (win % heads_per_rank))
    return (V.gather(0, best.unsqueeze(0)).squeeze(0),
            torch.where(win < 0, win, glob))


def _reduce_row(row: torch.Tensor, owner: torch.Tensor, heads_per_rank: int):
    """Combine one step's (2, n_keys) row and its owner across TP ranks.

    Means add then divide, maxima take a maximum: each rank owns a disjoint
    slice of the query heads. Done per step because selection must run on the
    whole row, not this rank's part of it.

    The owner needs more than a max. Each rank's `owner` holds
    ``layer * heads_per_rank + local_head``, which means nothing outside that
    rank -- rank 1's head 0 is global head `heads_per_rank`. So all-gather the
    (value, owner) pairs, take the argmax *rank* per position, and re-encode
    that rank's local head as a global one::

        global = layer * (heads_per_rank * world) + rank * heads_per_rank + local

    Reducing the value alone would leave `topk_head` naming rank 0's local
    winner even when another rank held the max, so the id and the value beside
    it would disagree.
    """
    import torch.distributed as dist

    from vllm.distributed.parallel_state import get_tp_group

    grp = get_tp_group().device_group
    world = dist.get_world_size(grp)

    out = row.clone()
    m = out[0].contiguous()
    dist.all_reduce(m, op=dist.ReduceOp.SUM, group=grp)
    out[0] = m / world

    vals = [torch.empty_like(row[1]) for _ in range(world)]
    owns = [torch.empty_like(owner) for _ in range(world)]
    dist.all_gather(vals, row[1].contiguous(), group=grp)
    dist.all_gather(owns, owner.contiguous(), group=grp)
    val, own = _resolve_owner(torch.stack(vals), torch.stack(owns).long(),
                              heads_per_rank)
    out[1] = val
    return out, own.to(owner.dtype)


class AttnConnector(KVConnectorBase_V1, SupportsHMA):
    """Exact per-decode-step attention capture.

    ``vllm_attn_connector.install_probe()`` must be called before the engine is
    built; the connector alone cannot see queries.

    Per decode token it stores the top ``top_pct`` percent of positions *within
    each ``chunk_size``-token block* of the prompt, as (position, score) pairs.
    Chunking guarantees every region of the prompt is represented, which a
    whole-prompt top-X% does not. ``top_pct=0`` emits the dense
    ``[steps, prompt_len]`` matrices instead, which is O(T) per step in memory
    and does not scale to long prompts; ``chunk_size=0`` selects globally.
    Configure with::

        --kv-transfer-config '{
          "kv_connector": "AttnConnector",
          "kv_connector_module_path": "vllm_attn_connector",
          "kv_role": "kv_producer",
          "kv_connector_extra_config": {
              "workflow_id": "w", "top_pct": 10, "chunk_size": 32
          }
        }'
    """

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole,
                 kv_cache_config: "KVCacheConfig"):
        super().__init__(vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config)
        extra = self._kv_transfer_config.kv_connector_extra_config or {}
        self._scheduler: _SchedulerSide | None = None
        self._worker: _WorkerSide | None = None
        if role == KVConnectorRole.SCHEDULER:
            self._scheduler = _SchedulerSide()
        else:
            parent = extra.get("workflow_id")
            wf = f"vllm-attn-{uuid.uuid4().hex[:12]}"
            logger.info("attn_connector: workflow_id=%s parent=%s", wf, parent)
            # Size the probe's buffers before anything runs a forward. They
            # must exist before the first CUDA graph is captured, because each
            # graph records the buffer's address; reallocating later would
            # leave already-captured graphs writing somewhere nobody reads.
            REGISTRY.max_rows = int(
                getattr(vllm_config.scheduler_config, "max_num_seqs", 0) or 256)
            self._worker = _WorkerSide(
                kv_cache_config, conf=_model_conf(vllm_config),
                tp_size=vllm_config.parallel_config.tensor_parallel_size,
                workflow_id=wf, parent_workflow_id=parent,
                max_steps=int(extra.get("max_steps", DEFAULT_MAX_STEPS)),
                top_pct=float(extra.get("top_pct", DEFAULT_TOP_PCT)),
                chunk=int(extra.get("chunk_size", DEFAULT_CHUNK)),
                out_dir=str(extra.get("out_dir", DEFAULT_OUT_DIR)),
                checksum=bool(extra.get("checksum", True)),
                layer_stats=bool(extra.get("layer_stats", False)),
            )

    @property
    def requires_kv_delivery(self) -> bool:
        return False

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: "VllmConfig") -> str | None:
        return None

    # -- scheduler side: pure delegation -------------------------------- #

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        return 0, False

    def update_state_after_alloc(self, request, blocks: "KVCacheBlocks", n_external):
        return

    def build_connector_meta(self, scheduler_output: "SchedulerOutput") -> KVConnectorMetadata:
        return self._scheduler.build_connector_meta(scheduler_output)

    def request_finished(self, request, block_ids):
        return self._scheduler.request_finished(request)

    def request_finished_all_groups(self, request, block_ids):
        # vLLM calls whichever of the two the engine version defines.
        return self._scheduler.request_finished(request)

    def update_connector_output(self, connector_output: "KVConnectorOutput") -> None:
        self._scheduler.update_connector_output(connector_output)

    def has_pending_push_work(self) -> bool:
        return self._scheduler.has_pending_push_work()

    # -- worker side ----------------------------------------------------- #

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        self._worker.register_kv_caches(kv_caches)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        """Emit finished requests. Runs on every path, including no-forward."""
        meta = self._get_connector_metadata()
        if isinstance(meta, AttnMetadata):
            self._worker.emit_finished(meta)

    def wait_for_save(self) -> None:
        """Score this step's decode tokens. Only called when a forward happened."""
        meta = self._get_connector_metadata()
        try:
            if isinstance(meta, AttnMetadata):
                self._worker.score_step(meta)
        finally:
            REGISTRY.begin_step()

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs: Any) -> None:
        return

    def build_connector_worker_meta(self) -> KVConnectorWorkerMetadata | None:
        return self._worker.take_emitted()

    def shutdown(self) -> None:
        if self._worker is not None:
            self._worker.shutdown()


__all__ = ["AttnConnector", "install"]
