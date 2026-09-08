# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Speculator for the nanoVLM MTP heads on a GraniteForDocling target.

Subclasses ``AutoRegressiveSpeculator``. The draft loop is deliberately NOT the
upstream shape ("one prefill pass for head 0, then one single-token step per
extra head"). It reproduces ``VisionLanguageModel._generate_speculative``, the
PyTorch reference that measured 66-87% per-head acceptance:

    every round, for head i in 0..K-1:
        run head i over the WHOLE newly-committed chunk (dense KV for head i),
        with the intervening token shifted i slots ahead -- the tail of each
        request is filled from THIS round's earlier drafts -- and a ZERO
        embedding wherever the intervening token lies inside the prefill,
        matching training, where the prompt's labels are -100;
        head i's full-span output is the hidden input of head i+1;
        head i's draft is sampled from its last slot.

Why the upstream shape is wrong for this model, measured:

* Upstream reuses ONE spec layer K times, so one dense KV cache serves every
  step. Our K heads are distinct modules with distinct caches. Under the
  upstream loop only head 0 ever sees a prefill; heads 1..4 write a single
  entry per round and attend over a cache that is otherwise zeros. Their
  conditional acceptance came out flat at 42-46% against head 0's 60%.
* Upstream embeds the image placeholder ids over the prompt. Training zeroed
  them. That pollutes exactly head 0's cache -- the head that was 27 points
  below its PyTorch figure.

All K passes reuse the TARGET's attention metadata and slot mappings: every
head runs over the same slots at the same positions, each into its own KV
layer, so the metadata is identical and the draft KV is dense by construction.
That also removes the need for per-step draft attention metadata entirely.

**The draft's attention geometry differs from the target's, and vLLM's
attention metadata builder does not know that.** The target is GQA (16 query
heads over 4 KV heads); the trained block is full MHA (16 over 16). The
FlashAttention-3 builder takes ``num_heads_kv`` from ``vllm_config.model_config``
-- the TARGET's -- and bakes it into the split-KV tile schedule
(``get_scheduler_metadata``). Upstream's shortcut of reusing the target's
attention metadata for the draft is therefore only valid when the draft copies
the target's geometry (EAGLE). For us it meant every short-query/long-KV step
-- prefix-cache-hit tails, chunked prefill, multi-request decode -- ran the
draft's attention against a schedule built for 4 KV heads: an illegal memory
access in ``flash_fwd_combine`` when it landed outside the buffers, silently
corrupted attention (acceptance 88% -> ~38%) when it did not. Single-shot
prefill uses one split, which is why batch 1 looked fine.

So the speculator builds its OWN attention groups from a VllmConfig whose
model_config is the DRAFT's, and every head pass -- eager or captured -- runs on
metadata built by those groups, refreshed into their static buffers each round.

CUDA graphs: one decode-style graph manager PER HEAD (decode_query_len = K+1),
each captured with its head pinned and with draft-built attention metadata. A
single shared graph cannot express "a different module per step" -- the index
is frozen at capture -- which is what forced the first working version to run
eager (87.8 tok/s vs 388.9 with graphs).

``load_draft_model`` wraps ``target.mtp`` (no second architecture) and
shares the target's tied ``embed_tokens`` / ``lm_head``. Draft Attention
is enabled after the target KV snapshot.
"""

from __future__ import annotations

import copy
import os
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.config.model import ModelConfig
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.v1.worker.gpu.attn_utils import (
    build_attn_metadata,
    build_slot_mappings_by_layer,
    init_attn_backend,
)
from vllm.v1.worker.gpu.cudagraph_utils import (
    AttentionStatePair,
    BatchExecutionDescriptor,
    get_uniform_token_count,
)
from vllm.v1.worker.gpu.dp_utils import dispatch_cg_and_sync_dp
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.autoregressive.cudagraph_utils import (
    DecodeSpeculatorCudaGraphManager,
)
from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
    AutoRegressiveSpeculator,
    prepare_prefill_inputs,
)

logger = init_logger(__name__)

# NANOVLM_MTP_TRACE=1 prints one line per propose() with the batch geometry the
# draft actually sees. Diagnostic only: it forces a device sync per step.
_TRACE = os.environ.get("NANOVLM_MTP_TRACE", "0") == "1"
# NANOVLM_MTP_DUMP=<dir>: save head-0 inputs/outputs per position per step so a
# chunked prefill can be diffed against a one-shot one. Diagnostic only.
_DUMP = os.environ.get("NANOVLM_MTP_DUMP", "")
# NANOVLM_MTP_SELFCHECK=1: after every FULL-graph round, re-run the identical
# round eagerly on the same buffers and report the first thing that differs.
# Diagnostic only (doubles draft work, syncs).
_SELFCHECK = os.environ.get("NANOVLM_MTP_SELFCHECK", "0") == "1"
# NANOVLM_MTP_TIMING=1: host wall-clock per propose() phase (CPU enqueue time,
# no syncs -- this is what sits on the critical path in front of each step),
# printed as running averages every 50 real calls.
_TIMING = os.environ.get("NANOVLM_MTP_TIMING", "0") == "1"


class _MTPAttnModelConfig(ModelConfig):
    """``ModelConfig`` view reporting MTP MHA geometry, not the VLM's GQA.

    Built by ``_mtp_model_config`` below, never constructed directly.
    """

    def get_total_num_kv_heads(self) -> int:
        return self._mtp_num_heads

    def get_num_kv_heads(self, parallel_config) -> int:
        return max(1, self._mtp_num_heads // parallel_config.tensor_parallel_size)

    def get_num_attention_heads(self, parallel_config) -> int:
        return self._mtp_num_heads // parallel_config.tensor_parallel_size

    def get_head_size(self) -> int:
        return self._mtp_head_size


def _mtp_model_config(base: ModelConfig, num_heads: int, hidden: int) -> ModelConfig:
    """Shallow-copy ``base`` and swap in the MTP-geometry class.

    ``ModelConfig`` and ``VllmConfig`` are pydantic dataclasses, so neither a
    subclass constructor nor ``dataclasses.replace`` can carry a proxy: pydantic
    revalidates every field and rejects anything that is not a real
    ``ModelConfig``. Copying the instance and reassigning ``__class__`` keeps the
    validated field values, satisfies ``isinstance`` checks in the attention
    backends, and overrides only the four geometry accessors.
    """
    view = copy.copy(base)
    view.__class__ = _MTPAttnModelConfig
    object.__setattr__(view, "_mtp_num_heads", int(num_heads))
    object.__setattr__(view, "_mtp_head_size", int(hidden) // int(num_heads))
    return view


def _resolve_embed_tokens(language_model, target_model):
    for holder in (language_model, target_model):
        if holder is None:
            continue
        for path in (
            "embed_tokens",
            "model.embed_tokens",
            "text_model.embed_tokens",
            "model.text_model.embed_tokens",
        ):
            mod = holder
            for attr in path.split("."):
                mod = getattr(mod, attr, None)
                if mod is None:
                    break
            if mod is not None:
                return mod
    return None


def _resolve_lm_head(language_model, target_model):
    for holder in (target_model, language_model):
        if holder is not None and getattr(holder, "lm_head", None) is not None:
            return holder.lm_head
    return None


class GraniteDoclingMTPSpeculator(AutoRegressiveSpeculator):
    """MTP speculator that drives K depth-specific heads, reference-faithfully."""

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        # inputs_embeds for the head currently running (static, for graphs).
        self.head_embeds = torch.zeros(
            self.max_num_tokens, self.hidden_size, dtype=self.dtype, device=device
        )
        # Per-batch-index prefill boundary: the slot where generation starts.
        # Same meaning as the reference's `prefill_len`.
        self.prefill_len = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=device
        )
        self.prefill_len_cpu = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, pin_memory=torch.cuda.is_available()
        )

    # ---- model loading -------------------------------------------------

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        draft = getattr(target_model, "mtp", None)
        if draft is None:
            raise AttributeError(
                "granite_for_docling MTP: target.mtp is missing; "
                "set GraniteForDoclingConfig.num_mtp_layers > 0"
            )
        # Attention is created here, after the target KV snapshot, so MTP
        # layers are counted as draft cache only.
        draft.enable_draft(self.vllm_config)

        language_model = (
            target_model.get_language_model()
            if hasattr(target_model, "get_language_model")
            else target_model
        )

        embed = _resolve_embed_tokens(language_model, target_model)
        if embed is None:
            raise AttributeError(
                "granite_for_docling MTP: could not locate the target embed_tokens"
            )
        draft.embed_tokens = embed

        lm_head = _resolve_lm_head(language_model, target_model)
        if lm_head is None:
            raise AttributeError(
                "granite_for_docling MTP: could not locate the target lm_head"
            )
        draft.lm_head = lm_head

        logger.info(
            "granite_for_docling MTP: using %d heads on the target; "
            "embed_tokens and lm_head shared.",
            len(draft.blocks),
        )
        self._maybe_load_draft_vocab(draft, lm_head)
        return draft

    def _maybe_load_draft_vocab(self, draft, lm_head) -> None:
        """Reduced draft vocabulary: config ``draft_vocab_file`` (relative to
        the draft model dir), overridable with NANOVLM_MTP_DRAFT_VOCAB=<path>
        or =0 to disable for an A/B."""
        import json

        target_cfg = self.vllm_config.model_config
        path = os.environ.get("NANOVLM_MTP_DRAFT_VOCAB")
        if path is None:
            path = getattr(target_cfg.hf_config, "mtp_draft_vocab_file", None)
            if path is None:
                path = getattr(target_cfg.hf_config, "draft_vocab_file", None)
        if not path or path == "0":
            logger.info(
                "granite_for_docling MTP: draft argmax over the full vocabulary"
            )
            return
        if not os.path.isabs(path):
            path = os.path.join(target_cfg.model, path)
        with open(path) as f:
            spec = json.load(f)
        ids = torch.tensor(spec["ids"], dtype=torch.int64)
        assert ids.numel() == ids.unique().numel(), "draft vocab has duplicate ids"
        draft.set_draft_vocab(ids, lm_head.weight.data)
        logger.info(
            "granite_for_docling MTP: draft argmax restricted to %d of %d vocab rows "
            "(corpus coverage %.3f%%, %s)",
            ids.numel(),
            lm_head.weight.shape[0],
            100 * spec.get("coverage", float("nan")),
            path,
        )

    def _greedy_sample_draft(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.model.draft_vocab_ids is not None:
            return self.model.draft_argmax(hidden_states)
        return super()._greedy_sample_draft(hidden_states)

    # ---- attention: draft-geometry metadata builders --------------------

    def set_attn(self, model_state, kv_cache_config, block_tables) -> None:
        self.model_state = model_state
        self.kv_cache_config = kv_cache_config
        self.block_tables = block_tables
        # FA3 builders read num_heads from vllm_config.model_config — the
        # target's GQA (16/4). MTP blocks are full MHA. Wrap the config so the
        # split-KV schedule matches the draft geometry.
        hf = self.vllm_config.model_config.hf_config
        text = getattr(hf, "text_config", hf)
        nhead = int(hf.mtp_num_attention_heads or text.num_attention_heads)
        hidden = int(text.hidden_size)
        draft_vllm_config = copy.copy(self.vllm_config)
        object.__setattr__(
            draft_vllm_config,
            "model_config",
            _mtp_model_config(self.vllm_config.model_config, nhead, hidden),
        )
        self.attn_groups, _, _ = init_attn_backend(
            kv_cache_config,
            draft_vllm_config,
            self.device,
            active_layer_names=self.draft_attn_layer_names,
        )
        # FA3's ahead-of-time tile schedule lives in a per-builder static buffer
        # that must be recomputed on the host every step (get_scheduler_metadata
        # + copy + zero-tail: 4 launches per builder). With it off, FA3 computes
        # the schedule inside the kernel, so a captured draft graph is
        # self-contained: on full-graph steps it needs only the static
        # query_start_loc / seq_lens / block-table / slot-mapping buffers that
        # the runner already refreshes, and propose() skips the metadata build.
        for groups in self.attn_groups:
            for group in groups:
                for builder in group.metadata_builders:
                    if getattr(builder, "aot_schedule", False):
                        builder.aot_schedule = False
        logger.info(
            "granite_for_docling MTP: draft attention groups built for "
            "kv_heads=%d (target has %d)",
            draft_vllm_config.model_config.get_num_kv_heads(
                self.vllm_config.parallel_config
            ),
            self.vllm_config.model_config.get_num_kv_heads(
                self.vllm_config.parallel_config
            ),
        )

    def _copy_request_inputs(self, num_reqs, idx_mapping, temperature, seeds) -> None:
        if self.draft_logits is None:
            # Greedy drafts: temperature and seeds are read by gumbel sampling
            # only; the base copies them every step (two of them from host
            # memory). Keep just the batch-index mapping the heads need.
            self.idx_mapping[:num_reqs].copy_(idx_mapping)
            return
        super()._copy_request_inputs(num_reqs, idx_mapping, temperature, seeds)

    def _build_chunk_attn_metadata(
        self,
        input_batch: InputBatch,
        num_reqs: int,
        num_reqs_padded: int,
        num_tokens_padded: int,
        max_query_len: int,
    ):
        """Draft attention metadata for the CURRENT chunk, built by the draft's
        own groups. Also refreshes those groups' static metadata buffers, which
        is what a captured FULL graph reads on replay."""
        # Slot mappings are plain slot indices (geometry-independent), so reuse
        # the ones the runner computed for the target THIS step from the
        # target's complete `positions`. Do NOT recompute them from the draft's
        # positions buffer: prepare_prefill_inputs fills it for only
        # query_len - num_rejected tokens per request, so the rejected slots
        # keep stale positions from earlier steps, and a recompute sends those
        # slots' KV writes to whatever position the stale value names --
        # silently overwriting real history (measured: 88% -> 74% acceptance).
        slot_mappings = self.block_tables.slot_mappings[:, :num_tokens_padded]
        qsl_np = input_batch.query_start_loc_np[: num_reqs + 1].astype(np.int32)
        if num_reqs_padded > num_reqs:
            qsl_np = np.concatenate(
                [
                    qsl_np,
                    np.full(num_reqs_padded - num_reqs, qsl_np[-1], dtype=np.int32),
                ]
            )
        attn_metadata = build_attn_metadata(
            attn_groups=self.attn_groups,
            num_reqs=num_reqs_padded,
            num_tokens=num_tokens_padded,
            query_start_loc_gpu=self.input_buffers.query_start_loc[
                : num_reqs_padded + 1
            ],
            query_start_loc_cpu=torch.from_numpy(qsl_np),
            max_query_len=int(max_query_len),
            seq_lens=self.input_buffers.seq_lens[:num_reqs_padded],
            max_seq_len=self.draft_max_seq_len,
            block_tables=[
                x[:num_reqs_padded] for x in self.block_tables.input_block_tables
            ],
            slot_mappings=slot_mappings,
            kv_cache_config=self.kv_cache_config,
        )
        return attn_metadata, build_slot_mappings_by_layer(
            slot_mappings, self.kv_cache_config
        )

    # ---- CUDA graphs: one decode-style manager per head -----------------

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        k = self.num_speculative_steps
        # Decode-style managers: their capture builds attention metadata from
        # OUR attn_groups (draft geometry). PIECEWISE is not supported for
        # these, same as upstream's draft decode.
        if cudagraph_mode.decode_mode() == CUDAGraphMode.FULL:
            decode_mode = CUDAGraphMode.FULL_DECODE_ONLY
        else:
            decode_mode = CUDAGraphMode.NONE
        # ONE manager, ONE graph per batch shape covering all K head passes:
        # every head runs over the same chunk with the same static buffers, and
        # the head index is a capture-time constant, so K graph launches (and
        # their inter-graph gaps) collapse into one.
        self._manager = DecodeSpeculatorCudaGraphManager(
            self.vllm_config, self.device, decode_mode, decode_query_len=k + 1
        )
        # Used for dispatch and by any base-class code expecting the attribute.
        self.prefill_cudagraph_manager = self._manager
        # Never single-step: every head runs over the full chunk.
        self.decode_cudagraph_manager = None

    def capture(
        self,
        attn_states: dict[BatchExecutionDescriptor, AttentionStatePair],
    ) -> None:
        logger.info(
            "Capturing MTP speculator graphs (%d heads)...", self.num_speculative_steps
        )
        self.last_token_indices.zero_()
        mgr = self._manager
        if mgr.use_breakable_cg:
            mgr.init_breakable_cg_runner(self.model)
        mgr.capture(
            self._all_heads_pass,
            self.model_state,
            self.input_buffers,
            self.block_tables,
            self.attn_groups,
            self.kv_cache_config,
            progress_bar_desc=(
                f"Capturing draft CUDA graphs ({self.num_speculative_steps} heads)"
            ),
        )

    # ---- draft loop ----------------------------------------------------

    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        num_tokens_across_dp: torch.Tensor | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        assert not aux_hidden_states, "MTP heads take the final hidden state only"
        k = self.num_speculative_steps
        num_tokens = input_batch.num_tokens_after_padding
        num_reqs = input_batch.num_reqs
        _t = time.perf_counter() if (_TIMING and not dummy_run) else None
        if _TRACE and not dummy_run:
            self._trace_step = getattr(self, "_trace_step", 0) + 1
            ns = num_sampled[:num_reqs].tolist()
            nr = num_rejected[:num_reqs].tolist()
            sm = next(iter(slot_mappings.values())) if slot_mappings else None
            smr = "-"
            if sm is not None:
                sm = sm[: input_batch.num_tokens]
                smr = f"[{int(sm.min())},{int(sm.max())}]"
            pos = input_batch.positions[: input_batch.num_tokens]
            ib = input_batch
            print(
                f"[mtp-trace step {self._trace_step}] reqs={num_reqs}"
                f" tokens={ib.num_tokens}/{num_tokens}"
                f" sched={ib.num_scheduled_tokens[:num_reqs].tolist()}"
                f" computed={ib.num_computed_tokens_np[:num_reqs].tolist()}"
                f" prefill_len={ib.prefill_len_np[:num_reqs].tolist()}"
                f" prefilling={ib.is_prefilling_np[:num_reqs].tolist()}"
                f" sampled={ns} rejected={nr} pos=[{int(pos.min())},{int(pos.max())}]"
                f" slots={smr}"
                f" layers_in_slot_map={len(slot_mappings) if slot_mappings else 0}",
                flush=True,
            )
        max_query_len = input_batch.num_scheduled_tokens.max()
        max_seq_len = input_batch.seq_lens_cpu_upper_bound[:num_reqs].max().item()
        self.draft_max_seq_len = min(max_seq_len + k, self.max_model_len)

        # Head 0 consumes the backbone's (post-final-norm) hidden state; each
        # later head consumes the previous head's output, written back below.
        self.hidden_states[:num_tokens].copy_(last_hidden_states)
        self._copy_request_inputs(num_reqs, input_batch.idx_mapping, temperature, seeds)
        # Same buffer prep as upstream: input_ids[t] = token(t+1), with the
        # last slot of each request set to the token the target just sampled.
        prepare_prefill_inputs(
            self.last_token_indices,
            self.current_draft_step,
            self.input_buffers,
            input_batch,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
            self.max_num_reqs,
        )
        if _t is not None:
            _t_prep = time.perf_counter()
        # Prefill boundary per batch index, CPU -> pinned -> device, no sync.
        self.prefill_len_cpu[:num_reqs].copy_(
            torch.from_numpy(input_batch.prefill_len_np[:num_reqs]).to(torch.int32)
        )
        self.prefill_len[:num_reqs].copy_(
            self.prefill_len_cpu[:num_reqs], non_blocking=True
        )

        uniform_token_count = get_uniform_token_count(
            num_reqs, input_batch.num_tokens, max_query_len
        )
        batch_desc, num_tokens_across_dp = dispatch_cg_and_sync_dp(
            self.prefill_cudagraph_manager,
            num_reqs,
            num_tokens,
            uniform_token_count,
            dp_size=self.dp_size,
            dp_rank=self.dp_rank,
            need_eager=is_profile,
        )
        self._prepare_eplb_forward(input_batch.num_tokens)

        # NOT the target's attn_metadata/slot_mappings: the draft's geometry
        # differs, so its metadata must come from its own builders. Only the
        # eager / piecewise path consumes the metadata object: a full-graph
        # replay reads the static buffers the runner refreshed for this step
        # (the draft graphs schedule attention in-kernel, see set_attn).
        if _t is not None:
            _t_disp = time.perf_counter()
        num_reqs_padded = batch_desc.num_reqs or num_reqs
        if (dummy_run and skip_attn_for_dummy_run) or not hasattr(self, "block_tables"):
            # Profile / dummy runs happen BEFORE set_attn() and ask us to skip
            # attention metadata (upstream does the same); the runner passes
            # None and the attention layers skip their kernel in that mode.
            draft_attn_metadata, draft_slot_mappings = attn_metadata, slot_mappings
        elif batch_desc.cg_mode == CUDAGraphMode.FULL and not _SELFCHECK:
            draft_attn_metadata, draft_slot_mappings = None, None
        else:
            draft_attn_metadata, draft_slot_mappings = self._build_chunk_attn_metadata(
                input_batch,
                num_reqs,
                num_reqs_padded,
                batch_desc.num_tokens,
                max_query_len,
            )

        if _SELFCHECK and batch_desc.cg_mode == CUDAGraphMode.FULL and not dummy_run:
            self._selfcheck_round(
                num_reqs,
                batch_desc,
                draft_attn_metadata,
                draft_slot_mappings,
                num_tokens_across_dp,
                mm_inputs,
            )
            return self.draft_tokens[:num_reqs]

        if _t is not None:
            _t_meta = time.perf_counter()
        if batch_desc.cg_mode == CUDAGraphMode.FULL:
            self._manager.run_fullgraph(batch_desc)
        else:
            self._all_heads_pass(
                num_reqs,
                batch_desc.num_tokens,
                draft_attn_metadata,
                draft_slot_mappings,
                num_tokens_across_dp,
                batch_desc.cg_mode,
                mm_inputs,
            )
        if _t is not None:
            _t_end = time.perf_counter()
            acc = getattr(self, "_timing", None)
            if acc is None:
                acc = self._timing = {
                    "n": 0,
                    "prep": 0.0,
                    "dispatch": 0.0,
                    "meta": 0.0,
                    "heads": 0.0,
                    "total": 0.0,
                    "full": 0,
                }
            acc["n"] += 1
            acc["prep"] += _t_prep - _t
            acc["dispatch"] += _t_disp - _t_prep
            acc["meta"] += _t_meta - _t_disp
            acc["heads"] += _t_end - _t_meta
            acc["total"] += _t_end - _t
            acc["full"] += int(batch_desc.cg_mode == CUDAGraphMode.FULL)
            if acc["n"] % 50 == 0:
                n = acc["n"]
                ms = {key: 1e3 * acc[key] / n for key in acc}
                print(
                    f"[mtp-timing] propose() host time over {n} calls"
                    f" (reqs={num_reqs}, K={k}, FULL-graph steps {acc['full']}/{n}):"
                    f" total {ms['total']:.2f} ms/step = prep {ms['prep']:.2f}"
                    f" + dispatch {ms['dispatch']:.2f}"
                    f" + attn-metadata {ms['meta']:.2f} + heads {ms['heads']:.2f}",
                    flush=True,
                )
        return self.draft_tokens[:num_reqs]

    def _selfcheck_round(
        self,
        num_reqs,
        batch_desc,
        attn_metadata,
        slot_mappings,
        num_tokens_across_dp,
        mm_inputs,
    ) -> None:
        nt = batch_desc.num_tokens
        nr = batch_desc.num_reqs or num_reqs
        hid0 = self.hidden_states[:nt].clone()
        dt0 = self.draft_tokens[:nr].clone()
        # Draft layer 0's KV cache restricted to this batch's blocks: the one
        # side effect the output comparison cannot see.
        attn0 = self.model.blocks[0].transformer_layer.attn
        kv = attn0.kv_cache[0]
        # Block-table rows are [num_reqs, max_blocks] with STALE entries past
        # each request's real block count; restrict to ids the tensor can hold.
        nb_dim = (
            0 if kv.shape[0] > 8 else 1
        )  # (num_blocks, ...) or (2, num_blocks, ...)
        blocks = self.block_tables.input_block_tables[0][:num_reqs].reshape(-1)
        blocks = blocks[(blocks > 0) & (blocks < kv.shape[nb_dim])].unique()

        def kv_at_blocks():
            return kv.index_select(nb_dim, blocks).clone()

        # --- graph replay ---
        self._manager.run_fullgraph(batch_desc)
        torch.accelerator.synchronize()
        dt_g = self.draft_tokens[:nr].clone()
        out_g = self.hidden_states[:nt].clone()
        emb_g = self.head_embeds[:nt].clone()
        kv_g = kv_at_blocks()
        # --- eager, same buffers, same metadata ---
        self.hidden_states[:nt].copy_(hid0)
        self.draft_tokens[:nr].copy_(dt0)
        self._all_heads_pass(
            nr,
            nt,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            CUDAGraphMode.NONE,
            mm_inputs,
        )
        torch.accelerator.synchronize()
        dt_e = self.draft_tokens[:nr].clone()
        out_e = self.hidden_states[:nt].clone()
        emb_e = self.head_embeds[:nt].clone()
        kv_e = kv_at_blocks()
        kv_bad = kv_g != kv_e
        kv_bad_elems = int(kv_bad.sum().item())
        kv_bad_blocks = (
            int(
                kv_bad.movedim(nb_dim, 0)
                .reshape(kv_bad.shape[nb_dim], -1)
                .any(dim=1)
                .sum()
                .item()
            )
            if blocks.numel()
            else 0  # a draining batch can reference no in-range block
        )
        n = getattr(self, "_sc_n", 0)
        self._sc_n = n + 1
        tok_mis = (dt_g[:num_reqs] != dt_e[:num_reqs]).sum().item()
        emb_d = (emb_g - emb_e).abs().max().item()
        out_d = (out_g - out_e).abs().reshape(nt, -1).max(dim=1).values
        per_req = []
        qsl = self.input_buffers.query_start_loc[: num_reqs + 1].tolist()
        for r in range(num_reqs):
            per_req.append(f"r{r}:{out_d[qsl[r] : qsl[r + 1]].max().item():.3g}")
        print(
            f"[mtp-selfcheck {n}] reqs={num_reqs}/{nr} tokens={nt}"
            f" draft-token mismatches={tok_mis}"
            f"  |embeds diff|={emb_d:.3g}  |out diff| per req: {' '.join(per_req)}"
            f"  seq_lens={self.input_buffers.seq_lens[:nr].tolist()}"
            f"  KV(layer0) graph!=eager: {kv_bad_elems} elems"
            f" in {kv_bad_blocks}/{len(blocks)} blocks",
            flush=True,
        )
        if n == 0 and tok_mis:
            bt = self.block_tables.input_block_tables[0][:nr, :4].tolist()
            print(
                f"[mtp-selfcheck] block_table[:, :4]={bt}"
                f" last_token_idx={self.last_token_indices[:nr].tolist()}",
                flush=True,
            )

    def _all_heads_pass(
        self,
        num_reqs: int,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        mm_inputs=None,
    ) -> None:
        """All K heads in sequence over the chunk; captured as ONE graph."""
        for head in range(self.num_speculative_steps):
            self.current_draft_step.fill_(head)
            self._head_pass(
                num_reqs,
                num_tokens,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp,
                cudagraph_runtime_mode,
                mm_inputs,
                head=head,
            )

    def _head_pass(
        self,
        num_reqs: int,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        mm_inputs=None,
        *,
        head: int,
    ) -> None:
        """One head over the whole chunk. Every op below reads/writes static
        buffers with shapes fixed by ``num_tokens``, so it is graph-capturable;
        the Python ``head`` is resolved at capture time, which is exactly what
        the per-head managers rely on."""
        k = self.num_speculative_steps
        self.model.set_spec_step(head)

        device = self.device
        last_idx = self.last_token_indices[:num_reqs]
        positions = self.input_buffers.positions[:num_tokens]
        idx_mapping = self.idx_mapping[:num_reqs]

        # Per-token request index and the request's last slot (graph-safe).
        qsl = self.input_buffers.query_start_loc[: num_reqs + 1]
        tok = torch.arange(num_tokens, dtype=qsl.dtype, device=device)
        req = (torch.searchsorted(qsl, tok, right=True) - 1).clamp_(0, num_reqs - 1)
        # The tail of each request is anchored on its last ACCEPTED slot, not
        # its last scheduled slot. vLLM pads every request's chunk to include
        # the positions REJECTED last round, and those slots still hold the
        # rejected draft tokens. Anchoring on qsl[req+1]-1 made head 1 read a
        # rejected token where it needed draft_0 -- conditional acceptance
        # collapsed to 13% (head 0 never touches the tail, so it was fine).
        # The reference never sees this because it crops rejected tokens out
        # of the chunk. Clamped so graph CAPTURE (zeroed indices, stale qsl)
        # stays in bounds; at replay the clamp is a no-op.
        anchor = self.last_token_indices[req].clamp_(0, num_tokens - 1)

        # Intervening token for head `head` at slot t is the token at
        # positions[t] + head + 1: the base buffer shifted `head` slots ahead,
        # and past the last accepted slot, this round's drafts in order
        # (draft_0 first). Mirrors intervening_for_chunk(). Slots beyond the
        # anchor are rejected/padding: their inputs are irrelevant (causal
        # attention, and they are recomputed next round).
        base = self.input_buffers.input_ids[:num_tokens]
        src = tok + head
        overflow = src - anchor  # > 0: past the last accepted slot
        from_base = base[torch.minimum(src, anchor)]
        draft_col = (overflow - 1).clamp(0, k - 1)
        from_drafts = self.draft_tokens[req, draft_col].to(base.dtype)
        ids = torch.where(overflow > 0, from_drafts, from_base)

        # Zero where the intervening token lies inside the prefill: training
        # masked those (-100 labels) and the reference does the same.
        in_prefill = (positions + (head + 1)) < self.prefill_len[req]
        embeds = self.model.embed_input_ids(ids)
        embeds = embeds.masked_fill(in_prefill.unsqueeze(-1), 0)
        self.head_embeds[:num_tokens].copy_(embeds)

        model_inputs = dict(
            input_ids=None,
            positions=positions,
            hidden_states=self.hidden_states[:num_tokens],
            inputs_embeds=self.head_embeds[:num_tokens],
        )
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            num_tokens_across_dp=num_tokens_across_dp,
            slot_mapping=slot_mappings,
            batch_descriptor=BatchDescriptor(num_tokens=num_tokens),
        ):
            if cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE:
                out = self._manager.run_pw_graph(self.model, model_inputs)
            else:
                out = self.model(**model_inputs)
        if isinstance(out, tuple):
            out = out[0]

        if _DUMP and head == 0 and not torch.cuda.is_current_stream_capturing():
            os.makedirs(_DUMP, exist_ok=True)
            n = getattr(self, "_dump_n", 0)
            self._dump_n = n + 1
            torch.save(
                {
                    "positions": positions.cpu(),
                    "ids": ids.cpu(),
                    "in_prefill": in_prefill.cpu(),
                    "req": req.cpu(),
                    "anchor": anchor.cpu(),
                    "hidden_in": self.hidden_states[:num_tokens].float().cpu(),
                    "embeds": self.head_embeds[:num_tokens].float().cpu(),
                    "out": out.float().cpu(),
                    "seq_lens": self.input_buffers.seq_lens[:num_reqs].cpu(),
                    "qsl": self.input_buffers.query_start_loc[: num_reqs + 1].cpu(),
                },
                os.path.join(_DUMP, f"step{n:04d}.pt"),
            )
        self.draft_tokens[:num_reqs, head] = self.sample_draft(
            out[last_idx],
            positions[last_idx],
            idx_mapping,
            self.temperature,
            self.seeds,
            self.current_draft_step,
            self.draft_logits,
        )
        # Chain: the next head's hidden input is this head's full-span output.
        self.hidden_states[:num_tokens].copy_(out)
