# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.config.vllm import VllmConfig
from vllm.model_executor.layers.attention import Attention
from vllm.v1.kv_cache_interface import KVCacheSpec, RSWASpec, get_kv_quant_mode


class RSWAAttention(Attention):
    """Attention layer that reports ``RSWASpec`` as its KV cache spec.

    Drop-in replacement for the standard ``Attention`` layer when the model is
    configured with Reference Sliding Window Attention (R-SWA,
    ``rswa_window > 0``). The actual masking logic lives in the attention
    backend (FlexAttention or FA4 mask_mod); this layer only overrides
    ``get_kv_cache_spec`` so the KV cache manager instantiates ``RSWAManager``
    (instead of ``FullAttentionManager``) and can therefore evict "gap" blocks
    to keep per-request KV memory bounded at O(prefix + window).
    """

    def __init__(self, *args, rswa_window: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._rswa_window = rswa_window

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        spec = super().get_kv_cache_spec(vllm_config)
        if spec is None:
            return None
        # Visibility extensions (read from the HF config; all default-off):
        # grammar anchors and strided far past keep positions beyond the
        # window visible, so gap-block eviction must be disabled for them;
        # attention sinks only shift the gap start past the sink band.
        hf_cfg = getattr(vllm_config.model_config, "hf_config", None)
        anchors_on = getattr(hf_cfg, "rswa_decode_anchors", "none") == "grammar"
        stride_on = int(getattr(hf_cfg, "rswa_stride", 0) or 0) > 0
        sink = int(getattr(hf_cfg, "rswa_sink_tokens", 0) or 0)
        # Default: blanket no-evict when anchors are on (keep all gap blocks).
        # Set rswa_anchor_evict=true for selective eviction of non-anchor gap
        # blocks. Stride keeps arbitrary far positions visible and requires
        # blanket no-evict.
        blanket = not bool(getattr(hf_cfg, "rswa_anchor_evict", False))
        anchor_evict = anchors_on and not blanket and not stride_on
        return RSWASpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            head_size_v=self.head_size_v,
            dtype=self.kv_cache_torch_dtype,
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
            rswa_window=self._rswa_window,
            rswa_no_evict=(anchors_on and blanket) or stride_on,
            rswa_sink=sink,
            rswa_anchor_evict=anchor_evict,
            rswa_model_path=vllm_config.model_config.model if anchor_evict else "",
            rswa_keep_all_locs=bool(getattr(hf_cfg, "rswa_keep_all_locs", False)),
            rswa_closed_trail_k=int(getattr(hf_cfg, "rswa_closed_trail_k", 0) or 0),
        )
