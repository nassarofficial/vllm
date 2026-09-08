# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multi-token prediction heads of ``GraniteForDoclingForConditionalGeneration``.

Not a registered vLLM architecture. ``GraniteForDoclingConfig.num_mtp_layers``
builds ``model.mtp`` on the target. The speculator wraps those weights after the
target KV snapshot so MTP attention is never counted as target cache.

Weight names match HF:
``mtp.blocks.{i}.{embed_norm,input_norm,proj,transformer_layer.*}``.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors

logger = init_logger(__name__)


def _text_config(hf_config):
    return getattr(hf_config, "text_config", None) or hf_config


class MTPTransformerBlock(nn.Module):
    """``nn.TransformerEncoderLayer(norm_first=True, gelu)`` with paged Attention.

    Linears and norms are created at target init (weight load). ``Attention`` is
    attached later by ``enable_draft_attention`` so it is not part of the
    target KV snapshot.
    """

    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        tcfg = _text_config(config)
        hidden_size = tcfg.hidden_size
        self.hidden_size = hidden_size
        self.total_num_heads = int(
            config.mtp_num_attention_heads or tcfg.num_attention_heads
        )
        ffn_dim = int(config.mtp_intermediate_size or tcfg.intermediate_size)
        self.head_dim = hidden_size // self.total_num_heads

        tp_size = get_tensor_model_parallel_world_size()
        assert self.total_num_heads % tp_size == 0, (
            f"MTP block has {self.total_num_heads} heads, not divisible by TP={tp_size}"
        )
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = self.num_heads

        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-5)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-5)
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.out_proj = RowParallelLinear(
            hidden_size,
            hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )
        self.linear1 = ColumnParallelLinear(
            hidden_size,
            ffn_dim,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.linear1",
        )
        self.linear2 = RowParallelLinear(
            ffn_dim,
            hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.linear2",
        )
        self.attn: Attention | None = None
        self._attn_prefix = f"{prefix}.attn"

    def enable_draft_attention(
        self,
        cache_config: CacheConfig | None,
        quant_config: QuantizationConfig | None,
    ) -> None:
        if self.attn is not None:
            return
        attn = Attention(
            self.num_heads,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=self._attn_prefix,
        )
        # This runs after the target is already on the GPU (MTP.enable_draft is
        # called post weight load), so nothing will move the layer for us.
        # `Attention` registers `_k_scale` / `_v_scale` as plain CPU tensors and
        # the Triton reshape_and_cache path takes them as raw pointers, which
        # rejects host memory ("cannot be accessed from Triton").
        self.attn = attn.to(self.qkv_proj.weight.device)

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        del positions  # trained block has no RoPE
        if self.attn is None:
            raise RuntimeError(
                "MTP attention is not enabled; speculator must call enable_draft"
            )
        residual = hidden_states
        x = self.norm1(hidden_states)
        qkv, _ = self.qkv_proj(x)
        q, k, v = qkv.split(
            [
                self.num_heads * self.head_dim,
                self.num_kv_heads * self.head_dim,
                self.num_kv_heads * self.head_dim,
            ],
            dim=-1,
        )
        attn_out = self.attn(q, k, v)
        del q, k, v
        hidden_states = residual + self.out_proj(attn_out)[0]
        x = self.norm2(hidden_states)
        x, _ = self.linear1(x)
        x = F.gelu(x)
        x, _ = self.linear2(x)
        return hidden_states + x


class MTPBlock(nn.Module):
    """One MTP head: RMSNorm fusion + transformer block."""

    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        tcfg = _text_config(config)
        eps = getattr(tcfg, "rms_norm_eps", 1e-5)
        self.embed_norm = RMSNorm(tcfg.hidden_size, eps=eps)
        self.input_norm = RMSNorm(tcfg.hidden_size, eps=eps)
        self.proj = ReplicatedLinear(
            tcfg.hidden_size * 2,
            tcfg.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.proj",
        )
        self.transformer_layer = MTPTransformerBlock(
            config, quant_config, f"{prefix}.transformer_layer"
        )

    def enable_draft_attention(
        self,
        cache_config: CacheConfig | None,
        quant_config: QuantizationConfig | None,
    ) -> None:
        self.transformer_layer.enable_draft_attention(cache_config, quant_config)

    def forward(
        self,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor,
    ) -> torch.Tensor:
        e = self.embed_norm(inputs_embeds)
        h = self.input_norm(previous_hidden_states)
        # Training / HF order: cat[hnorm(hidden), enorm(embed)].
        x, _ = self.proj(torch.cat([h, e], dim=-1))
        return self.transformer_layer(positions, x)


class MTP(nn.Module):
    """K-head MTP module attached to GraniteForDocling.

    Draft interface for the speculator."""

    has_own_embed_tokens = False
    draft_vocab_ids: torch.Tensor | None = None
    draft_head_weight: torch.Tensor | None = None

    def __init__(
        self,
        hf_config,
        *,
        vllm_config: VllmConfig,
        prefix: str = "mtp",
    ) -> None:
        super().__init__()
        n_heads = int(hf_config.num_mtp_layers)
        if n_heads <= 0:
            raise ValueError("MTP requires config.num_mtp_layers > 0")
        self.config = hf_config
        self.quant_config = vllm_config.quant_config
        self.num_heads = n_heads
        self.blocks = nn.ModuleList(
            [
                MTPBlock(hf_config, vllm_config.quant_config, f"{prefix}.blocks.{i}")
                for i in range(n_heads)
            ]
        )
        tcfg = _text_config(hf_config)
        logits_scaling = getattr(tcfg, "logits_scaling", 1.0) or 1.0
        self.logits_processor = LogitsProcessor(
            tcfg.vocab_size, scale=1.0 / logits_scaling
        )
        self.embed_tokens = None
        self.lm_head = None
        self._spec_step = 0
        self._draft_ready = False

    def enable_draft(self, vllm_config: VllmConfig) -> None:
        """Attach paged Attention after the target KV snapshot."""
        if self._draft_ready:
            return
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        for block in self.blocks:
            block.enable_draft_attention(cache_config, quant_config)
        self._draft_ready = True
        logger.info(
            "granite_for_docling mtp: enabled draft attention on %d heads",
            len(self.blocks),
        )

    def set_spec_step(self, step: int) -> None:
        self._spec_step = int(step)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.embed_tokens is None:
            raise RuntimeError("MTP embed_tokens is not shared from the target yet")
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int | None = None,
    ) -> torch.Tensor:
        del intermediate_tensors
        step = self._spec_step if spec_step_idx is None else spec_step_idx
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        return self.blocks[step % self.num_heads](
            positions, hidden_states, inputs_embeds
        )

    def compute_logits(
        self, hidden_states: torch.Tensor, spec_step_idx: int | None = None
    ):
        del spec_step_idx
        if self.lm_head is None:
            raise RuntimeError("MTP lm_head is not shared from the target yet")
        return self.logits_processor(self.lm_head, hidden_states)

    def set_draft_vocab(self, ids: torch.Tensor, lm_head_weight: torch.Tensor) -> None:
        ids = ids.to(lm_head_weight.device, torch.int64)
        self.draft_vocab_ids = ids
        self.draft_head_weight = lm_head_weight.index_select(0, ids).contiguous()

    def draft_argmax(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = torch.matmul(hidden_states, self.draft_head_weight.t())
        return self.draft_vocab_ids[logits.argmax(dim=-1)]

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Map HF ``mtp.blocks.{i}.*`` names onto the vLLM module tree."""
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        for name, weight in weights:
            qkv_mapped = _split_in_proj(name, weight)
            if qkv_mapped is not None:
                mapped, shards = qkv_mapped
                if mapped not in params:
                    continue
                param = params[mapped]
                for shard_id, shard_w in shards:
                    param.weight_loader(param, shard_w, shard_id)
                loaded.add(mapped)
                continue
            mapped = name.replace(
                "transformer_layer.self_attn.out_proj",
                "transformer_layer.out_proj",
            )
            if mapped.endswith(".bias") and mapped not in params:
                continue
            if mapped not in params:
                continue
            param = params[mapped]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, weight)
            loaded.add(mapped)
        return loaded


def _split_in_proj(
    name: str, weight: torch.Tensor
) -> tuple[str, list[tuple[str, torch.Tensor]]] | None:
    if "self_attn.in_proj_weight" in name:
        mapped = name.replace(
            "transformer_layer.self_attn.in_proj_weight",
            "transformer_layer.qkv_proj.weight",
        )
        q, k, v = weight.chunk(3, dim=0)
        return mapped, [("q", q), ("k", k), ("v", v)]
    if "self_attn.in_proj_bias" in name:
        mapped = name.replace(
            "transformer_layer.self_attn.in_proj_bias",
            "transformer_layer.qkv_proj.bias",
        )
        q, k, v = weight.chunk(3, dim=0)
        return mapped, [("q", q), ("k", k), ("v", v)]
    return None


def maybe_build_mtp(
    hf_config,
    vllm_config: VllmConfig,
    prefix: str = "mtp",
) -> MTP | None:
    if int(getattr(hf_config, "num_mtp_layers", 0)) <= 0:
        return None
    return MTP(hf_config, vllm_config=vllm_config, prefix=prefix)
