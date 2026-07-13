# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only GraniteDoclingHybrid VLM compatible with HuggingFace weights.

Architecture (see ``transformers.models.granite_docling_hybrid``):

* SigLIP-style vision encoder (``Idefics2VisionTransformer``) with
  intermediate features tapped at ``config.deepstack_visual_indexes``.
* Modality-projector connector implementing the ``pixel_shuffle_mlp_v2``
  variant: ``LayerNorm -> pixel_shuffle -> Linear -> + 2D sincos pos embed
  -> LayerNorm -> GELU -> Linear -> LayerNorm``. Per-tap ``DeepStack``
  mergers project intermediate ViT features to LM space.
* ``GraniteMoeHybrid`` text decoder. During prefill, DeepStack features
  are added residually to the hidden states at image-token positions
  after the LM decoder layers listed in ``config.deepstack_attn_layers``
  (slot ``i`` of ``deepstack_visual_indexes`` ↔ slot ``i`` of
  ``deepstack_attn_layers``). DeepStack injection is restricted to
  prefill — autoregressive decoding has no image tokens.
"""

from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, Literal, TypeAlias

import numpy as np
import torch
from torch import nn
from transformers import BatchFeature, Idefics3Processor

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateCopyFunc
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import ImageProcessorItems, MultiModalDataItems
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.sequence import IntermediateTensors
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .granitemoehybrid import GraniteMoeHybridForCausalLM, GraniteMoeHybridModel
from .idefics2_vision_model import (
    Idefics2VisionTransformer as _BaseIdefics3VisionTransformer,
)
from .interfaces import (
    HasInnerState,
    IsHybrid,
    MultiModalEmbeddings,
    SupportsLoRA,
    SupportsMambaPrefixCaching,
    SupportsMultiModal,
    SupportsPP,
    _require_is_multimodal,
)
from .utils import (
    AutoWeightsLoader,
    _merge_multimodal_embeddings,
    maybe_prefix,
)

logger = init_logger(__name__)


# -----------------------------------------------------------------------------
# Tensor schemas
# -----------------------------------------------------------------------------


class GraniteDoclingHybridImagePixelInputs(TensorSchema):
    """
    Dimensions:
        - bnp: Batch size * number of images * number of patches
        - c: Number of channels (3)
        - h: Height
        - w: Width
        - bn: Batch size * number of images
    """

    type: Literal["pixel_values"]
    pixel_values: Annotated[torch.Tensor, TensorShape("bnp", 3, "h", "w")]
    num_patches: Annotated[torch.Tensor, TensorShape("bn")]


class GraniteDoclingHybridImageEmbeddingInputs(TensorSchema):
    """
    Dimensions:
        - bn: Batch size * number of images
        - f: Image feature size
        - h: Hidden size of the language model (or hidden size *
              ``1 + num_deepstack_levels`` when DeepStack is enabled)
    """

    type: Literal["image_embeds"]
    data: Annotated[torch.Tensor, TensorShape("bn", "f", "h")]


ImageInputs: TypeAlias = (
    GraniteDoclingHybridImagePixelInputs | GraniteDoclingHybridImageEmbeddingInputs
)


# -----------------------------------------------------------------------------
# Processing
#
# ``GraniteDoclingHybridProcessor`` is a subclass of ``Idefics3Processor`` (lives
# in HF Transformers) that pairs with ``GotOcr2ImageProcessor`` to perform
# aspect-ratio-aware tiling. Compared to the stock Idefics3 path this means:
#
# * ``image_processor.size`` is ``{"height": H, "width": W}`` (not
#   ``{"longest_edge": L}``).
# * The image processor is parameterised by ``min_patches`` / ``max_patches``
#   and uses ``get_optimal_tiled_canvas`` to pick a grid.
# * The HF processor wraps the raw ``[total_patches, C, H, W]`` GotOcr2 output
#   into the canonical Idefics3 ``[batch, max_patches, C, H, W]`` 5D shape and
#   adds an explicit ``pixel_attention_mask`` over the patch axis.
#
# To stay independent of the stock Idefics3 vLLM module (which assumes
# ``longest_edge`` everywhere) we implement everything we need here directly.
# -----------------------------------------------------------------------------


def _get_optimal_tiled_canvas_grid(
    image_height: int,
    image_width: int,
    canvas_height: int,
    canvas_width: int,
    min_patches: int,
    max_patches: int,
) -> tuple[int, int]:
    """Replicate ``GotOcr2ImageProcessor.get_optimal_tiled_canvas`` with the
    same area-based tie-breaking. Returns ``(num_columns, num_rows)``.

    We prefer to use the function from HF Transformers when it is present so
    we exactly match the processor's tile count (area-tie-break included);
    fall back to a naive aspect-ratio search otherwise.
    """
    try:
        from transformers.models.got_ocr2.image_processing_got_ocr2 import (
            get_optimal_tiled_canvas,
        )
    except ImportError:
        get_optimal_tiled_canvas = None

    if get_optimal_tiled_canvas is not None:
        num_columns, num_rows = get_optimal_tiled_canvas(
            (image_height, image_width),
            (canvas_height, canvas_width),
            min_patches,
            max_patches,
        )
        return num_columns, num_rows

    image_ar = image_width / image_height
    best_cols, best_rows = 1, 1
    best_diff = float("inf")
    for cols in range(1, max_patches + 1):
        for rows in range(1, max_patches + 1):
            total = cols * rows
            if total < min_patches or total > max_patches:
                continue
            diff = abs(image_ar - cols / rows)
            if diff < best_diff or (
                diff == best_diff and total > best_cols * best_rows
            ):
                best_diff = diff
                best_cols, best_rows = cols, rows
    return best_cols, best_rows


class GraniteDoclingHybridProcessingInfo(BaseProcessingInfo):
    """Provides metadata about the HF processor / image processor.

    The HF processor class is the user's ``GraniteDoclingHybridProcessor``,
    a subclass of ``Idefics3Processor``; using ``Idefics3Processor`` as the
    type hint is safe and lets the processor be auto-resolved from the HF
    auto-loader.
    """

    def get_hf_processor(self, **kwargs: object) -> Idefics3Processor:
        return self.ctx.get_hf_processor(Idefics3Processor, **kwargs)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}

    # ---- image-grid computation (GotOcr2 crop_to_patches) ----

    def _get_image_size(self, image_processor) -> tuple[int, int]:
        size = image_processor.size
        return int(size["height"]), int(size["width"])

    def _get_grid_size(
        self,
        *,
        image_width: int,
        image_height: int,
        image_processor,
    ) -> tuple[int, int]:
        canvas_h, canvas_w = self._get_image_size(image_processor)
        max_patches = int(getattr(image_processor, "max_patches", 16))
        min_patches = int(getattr(image_processor, "min_patches", 1))
        return _get_optimal_tiled_canvas_grid(
            image_height=image_height,
            image_width=image_width,
            canvas_height=canvas_h,
            canvas_width=canvas_w,
            min_patches=min_patches,
            max_patches=max_patches,
        )

    def get_num_patches(
        self,
        *,
        image_width: int,
        image_height: int,
        processor: Idefics3Processor,
        mm_kwargs: Mapping[str, object],  # noqa: ARG002 — unused but kept for parity
    ) -> int:
        """Total patch count for a single image: ``rows*cols + 1`` thumbnail.

        For single-tile images (rows == cols == 1) GotOcr2 skips the
        thumbnail patch, so the count is just 1.
        """
        grid_w, grid_h = self._get_grid_size(
            image_width=image_width,
            image_height=image_height,
            image_processor=processor.image_processor,
        )
        if grid_w * grid_h <= 1:
            return 1
        return grid_w * grid_h + 1

    # ---- placeholder expansion ----

    def _get_image_token(
        self, processor: Idefics3Processor
    ) -> tuple[str, str, str]:
        return (
            processor.image_token,
            processor.fake_image_token,
            processor.global_image_tag,
        )

    def get_image_repl(
        self,
        *,
        image_width: int,
        image_height: int,
        processor: Idefics3Processor,
        mm_kwargs: Mapping[str, object],
    ) -> str:
        image_token, fake_image_token, global_img_token = self._get_image_token(
            processor
        )
        image_seq_len = processor.image_seq_len
        p_img = image_token * image_seq_len
        global_img_placeholder = fake_image_token + global_img_token + p_img

        grid_w, grid_h = self._get_grid_size(
            image_width=image_width,
            image_height=image_height,
            image_processor=processor.image_processor,
        )

        if grid_w * grid_h <= 1:
            return global_img_placeholder + fake_image_token

        tile_img_placeholder = fake_image_token + "<row_{n_h}_col_{n_w}>" + p_img
        tiles: list[str] = []
        for i in range(grid_h):
            for j in range(grid_w):
                tiles.append(tile_img_placeholder.format(n_h=i + 1, n_w=j + 1))
                if j == grid_w - 1:
                    tiles.append("\n")
        return "".join([*tiles, "\n", global_img_placeholder, fake_image_token])

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        processor: Idefics3Processor,
        mm_kwargs: Mapping[str, object],
    ) -> int:
        num_patches = self.get_num_patches(
            image_width=image_width,
            image_height=image_height,
            processor=processor,
            mm_kwargs=mm_kwargs,
        )
        return num_patches * processor.image_seq_len


class GraniteDoclingHybridDummyInputsBuilder(
    BaseDummyInputsBuilder[GraniteDoclingHybridProcessingInfo]
):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        processor = self.info.get_hf_processor()
        image_token, _, _ = self.info._get_image_token(processor)
        return image_token * num_images

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        image_processor = self.info.get_hf_processor().image_processor
        h, w = self.info._get_image_size(image_processor)
        # Worst-case tiling for memory profiling: a max_patches x 1 grid plus
        # the thumbnail tile, i.e. the maximum number of image tokens per item.
        max_patches = int(getattr(image_processor, "max_patches", 16))
        image_overrides = mm_options.get("image") if mm_options else None
        return {
            "image": self._get_dummy_images(
                width=w * max_patches,
                height=h,
                num_images=num_images,
                overrides=image_overrides,
            )
        }


class GraniteDoclingHybridMultiModalProcessor(
    BaseMultiModalProcessor[GraniteDoclingHybridProcessingInfo]
):
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        # Text-only — bypass the composite processor and go straight to the
        # tokenizer (consistent with Idefics3MultiModalProcessor).
        if not (images := mm_data.get("images", [])):
            prompt_ids = self.info.get_tokenizer().encode(prompt)
            prompt_ids = self._apply_hf_processor_tokens_only(prompt_ids)
            return BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")

        # PIL-backend processors expect channels-last numpy input; torch-backend
        # (fast) processors handle layout themselves (see vllm#48467).
        image_processor = self.info.get_hf_processor().image_processor
        if getattr(image_processor, "backend", "pil") == "pil":
            mm_kwargs = {"input_data_format": "channels_last", **mm_kwargs}
        processed_outputs = super()._call_hf_processor(
            prompt, mm_data, mm_kwargs, tok_kwargs
        )

        hf_processor = self.info.get_hf_processor(**mm_kwargs)

        pv = processed_outputs["pixel_values"]

        # ``GraniteDoclingHybridProcessor`` reshapes GotOcr2's flat
        # ``[total_patches, C, H, W]`` into ``[batch, max_patches, C, H, W]``
        # and emits ``pixel_attention_mask`` as ``[batch, max_patches]``.
        # In that case ``num_patches`` isn't in the BatchFeature — we
        # recompute it from the image sizes for vLLM's per-image splitting.
        # When wrapping the raw GotOcr2 processor directly (no Idefics3
        # post-processing) the output is the 4D form with ``num_patches`` in
        # the BatchFeature; we handle that too.
        num_patches_raw = processed_outputs.get("num_patches")
        if num_patches_raw is not None:
            num_patches = (
                num_patches_raw.long()
                if isinstance(num_patches_raw, torch.Tensor)
                else torch.tensor(num_patches_raw, dtype=torch.long)
            )
        else:
            mm_items = self.info.parse_mm_data({"image": images}, validate=False)
            parsed_images = mm_items.get_items("image", ImageProcessorItems)
            num_patches = torch.tensor(
                [
                    self.info.get_num_patches(
                        image_width=parsed_images.get_image_size(i).width,
                        image_height=parsed_images.get_image_size(i).height,
                        processor=hf_processor,
                        mm_kwargs=mm_kwargs,
                    )
                    for i in range(len(parsed_images))
                ],
                dtype=torch.long,
            )
        processed_outputs["num_patches"] = num_patches

        # Normalise pixel_values to a flat ``[total_patches, C, H, W]`` so the
        # downstream pipeline doesn't need to special-case Idefics3-shaped vs
        # GotOcr2-shaped outputs. If the processor returned a padded 5D
        # tensor we drop the pad slices using ``pixel_attention_mask``
        # (preferred) or the all-zero heuristic.
        if pv.ndim == 5:
            existing_pam = processed_outputs.get("pixel_attention_mask")
            if existing_pam is not None and existing_pam.dim() == 2:
                keep = existing_pam.view(-1).bool()
                pv = pv.reshape(-1, *pv.shape[2:])[keep].contiguous()
            else:
                pv_flat = pv.reshape(-1, *pv.shape[2:])
                keep = pv_flat.sum(dim=(-1, -2, -3)) != 0
                if not keep.all():
                    keep = (pv_flat == 0).sum(dim=(-1, -2, -3)) != pv_flat[
                        0
                    ].numel()
                pv = pv_flat[keep].contiguous()
            processed_outputs["pixel_values"] = pv
        elif pv.ndim == 4:
            # Already flat — nothing to do.
            pass
        else:
            raise ValueError(
                f"Unsupported pixel_values rank {pv.ndim}; expected 4 or 5."
            )

        # GotOcr2 crops each tile to exactly the ViT resolution — every pixel
        # is valid — so no pixel_attention_mask is emitted (the vision tower
        # builds an all-ones patch mask internally when given None).
        processed_outputs.pop("pixel_attention_mask", None)

        return processed_outputs

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        num_patches = hf_inputs.get("num_patches", torch.empty(0))
        return dict(
            pixel_values=MultiModalFieldConfig.flat_from_sizes("image", num_patches),
            image_embeds=MultiModalFieldConfig.batched("image"),
            num_patches=MultiModalFieldConfig.batched("image"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        image_token, _, _ = self.info._get_image_token(hf_processor)

        def get_replacement(item_idx: int) -> PromptUpdateDetails:
            images = mm_items.get_items("image", ImageProcessorItems)
            image_size = images.get_image_size(item_idx)
            image_repl = self.info.get_image_repl(
                image_width=image_size.width,
                image_height=image_size.height,
                processor=hf_processor,
                mm_kwargs=hf_processor_mm_kwargs,
            )
            return PromptUpdateDetails.select_text(
                image_repl, embed_text=image_token
            )

        return [
            PromptReplacement(
                modality="image",
                target=image_token,
                replacement=get_replacement,
            )
        ]


# -----------------------------------------------------------------------------
# Vision encoder with intermediate-feature taps for DeepStack
# -----------------------------------------------------------------------------


class GraniteDoclingHybridVisionTransformer(_BaseIdefics3VisionTransformer):
    """SigLIP-style vision encoder that returns intermediate features.

    Inherits from the standard Idefics2 vision transformer used by Idefics3 and
    overrides ``forward`` to also emit the residual-stream output of each block
    whose index appears in ``deepstack_visual_indexes`` — taps are taken
    BEFORE the post-LayerNorm to match the HF reference (``capture_outputs``
    with ``tie_last_hidden_states=False``).
    """

    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        *,
        num_hidden_layers_override: int | None = None,
        require_post_norm: bool = True,
        deepstack_visual_indexes: Sequence[int] | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            config,
            quant_config=quant_config,
            num_hidden_layers_override=num_hidden_layers_override,
            require_post_norm=require_post_norm,
            prefix=prefix,
        )
        self.deepstack_visual_indexes: list[int] = list(deepstack_visual_indexes or [])
        # Set form is used for O(1) check inside the encoder loop.
        self._deepstack_visual_index_set = set(self.deepstack_visual_indexes)

    def forward(
        self,
        pixel_values: torch.Tensor,
        patch_attention_mask: torch.BoolTensor | None = None,
        tgt_sizes: torch.IntTensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if patch_attention_mask is None:
            # GotOcr2 tiles are always fully valid; the embeddings only need
            # an all-ones patch-level mask for position ids.
            patch_attention_mask = torch.ones(
                (
                    pixel_values.size(0),
                    pixel_values.size(2) // self.config.patch_size,
                    pixel_values.size(3) // self.config.patch_size,
                ),
                dtype=torch.bool,
                device=pixel_values.device,
            )

        hidden_states = self.embeddings(
            pixel_values=pixel_values,
            patch_attention_mask=patch_attention_mask,
            tgt_sizes=tgt_sizes,
        )

        intermediate_features: list[torch.Tensor] = []
        # We bypass the encoder's batched ``forward`` so we can tap residual
        # stream values mid-stack. ``use_data_parallel`` ViTs use the sharded
        # path inside ``Idefics2VisionTransformer.forward``; DeepStack is not
        # exercised in that configuration and would require splitting the tap
        # outputs across shards, so we keep the simple per-layer loop here.
        for i, encoder_layer in enumerate(self.encoder.layers):
            hidden_states = encoder_layer(hidden_states)
            if i in self._deepstack_visual_index_set:
                intermediate_features.append(hidden_states)

        last_hidden_state = self.post_layernorm(hidden_states)
        return last_hidden_state, intermediate_features


# -----------------------------------------------------------------------------
# Connector (modality projector) — pixel_shuffle{,_mlp,_mlp_v2}
# -----------------------------------------------------------------------------


def _build_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
    """MAE-style 2D sin-cos positional embedding of shape ``[grid_size**2, D]``.

    Mirrors the construction used by the HF / nanoVLM reference
    (``pixel_shuffle_mlp_v2``) so a trained checkpoint loads byte-for-byte.
    """
    if embed_dim % 4 != 0:
        raise ValueError(
            "embed_dim must be divisible by 4 for 2D sincos positional embeddings."
        )

    def _1d_sincos(dim: int, pos: np.ndarray) -> np.ndarray:
        omega = np.arange(dim // 2, dtype=np.float32)
        omega /= dim / 2.0
        omega = 1.0 / 10000**omega
        pos = pos.reshape(-1)
        out = np.einsum("m,d->md", pos, omega)
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)

    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])

    emb_h = _1d_sincos(embed_dim // 2, grid[0])
    emb_w = _1d_sincos(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return torch.from_numpy(emb).float()


def _pixel_shuffle(x: torch.Tensor, scale_factor: int) -> torch.Tensor:
    """Spatial-to-channel pixel shuffle matching the HF / nanoVLM convention."""
    bsz, seq, embed_dim = x.size()
    height = width = int(seq**0.5)
    x = x.view(bsz, height, width, embed_dim)
    x = x.view(bsz, height, width // scale_factor, embed_dim * scale_factor)
    x = x.permute(0, 2, 1, 3)
    x = x.reshape(
        bsz,
        width // scale_factor,
        height // scale_factor,
        embed_dim * (scale_factor**2),
    )
    x = x.permute(0, 2, 1, 3)
    x = x.reshape(
        bsz, seq // (scale_factor**2), embed_dim * (scale_factor**2)
    )
    return x


class GraniteDoclingHybridDeepStackMerger(nn.Module):
    """Per-tap projector mapping intermediate ViT features to LM space.

    ``pixel_shuffle -> LayerNorm -> Linear -> GELU -> Linear``. One instance
    is created per entry of ``config.deepstack_visual_indexes``.
    """

    def __init__(
        self,
        vision_hidden_size: int,
        scale_factor: int,
        text_hidden_size: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.scale_factor = scale_factor
        merged_dim = vision_hidden_size * (scale_factor**2)
        # The DeepStack mergers are small enough that the linears stay
        # un-sharded; using ``ReplicatedLinear`` keeps weights identical
        # across tensor-parallel ranks and matches the HF reference.
        self.norm = nn.LayerNorm(merged_dim)
        self.fc1 = ReplicatedLinear(
            merged_dim,
            text_hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "fc1"),
        )
        self.act = nn.GELU()
        self.fc2 = ReplicatedLinear(
            text_hidden_size,
            text_hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "fc2"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _pixel_shuffle(x.contiguous(), self.scale_factor)
        x = self.norm(x)
        x, _ = self.fc1(x)
        x = self.act(x)
        x, _ = self.fc2(x)
        return x


class GraniteDoclingHybridSimpleMLP(nn.Module):
    """Main modality projection: ``Linear(vit_dim * sf^2 -> lm_dim)``."""

    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        in_size = config.vision_config.hidden_size * (config.scale_factor**2)
        out_size = config.text_config.hidden_size
        self.proj = ReplicatedLinear(
            in_size,
            out_size,
            bias=False,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "proj"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.proj(x)
        return out


class GraniteDoclingHybridConnector(nn.Module):
    """Modality projector matching nanoVLM's three ``mp_pooling_mode`` variants.

    * ``pixel_shuffle`` — ``pixel_shuffle -> proj``.
    * ``pixel_shuffle_mlp`` — adds ``-> GELU -> mlp_fc2`` (channel mix).
    * ``pixel_shuffle_mlp_v2`` — full MiniCPM-style resampler:
      ``ln_in -> pixel_shuffle -> proj -> + pos_embed_2d -> ln_mid -> GELU ->
      mlp_fc2 -> ln_out``. ``pos_embed_2d`` is a non-persistent 2D sincos
      buffer rebuilt at construction.

    When ``config.use_deepstack`` is True, ``deepstack_mergers`` is an
    ``nn.ModuleList`` of [`GraniteDoclingHybridDeepStackMerger`] — one per
    ViT tap. DeepStack runs in parallel with the main pooling path.
    """

    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.scale_factor = config.scale_factor
        self.pooling_mode = getattr(config, "mp_pooling_mode", "pixel_shuffle")
        if self.pooling_mode not in (
            "pixel_shuffle",
            "pixel_shuffle_mlp",
            "pixel_shuffle_mlp_v2",
        ):
            raise ValueError(
                f"Unknown mp_pooling_mode={self.pooling_mode!r}. Supported: "
                "'pixel_shuffle', 'pixel_shuffle_mlp', 'pixel_shuffle_mlp_v2'."
            )

        text_hidden_size = config.text_config.hidden_size
        vision_hidden_size = config.vision_config.hidden_size

        self.modality_projection = GraniteDoclingHybridSimpleMLP(
            config,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "modality_projection"),
        )

        # --- pixel_shuffle_mlp{,_v2} extras ---
        self.mlp_fc2: nn.Module | None = None
        if self.pooling_mode in ("pixel_shuffle_mlp", "pixel_shuffle_mlp_v2"):
            self.mlp_fc2 = ReplicatedLinear(
                text_hidden_size,
                text_hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "mlp_fc2"),
            )

        self.ln_in: nn.LayerNorm | None = None
        self.ln_mid: nn.LayerNorm | None = None
        self.ln_out: nn.LayerNorm | None = None
        if self.pooling_mode == "pixel_shuffle_mlp_v2":
            self.ln_in = nn.LayerNorm(vision_hidden_size)
            self.ln_mid = nn.LayerNorm(text_hidden_size)
            self.ln_out = nn.LayerNorm(text_hidden_size)

            tokens_per_tile = (
                (config.vision_config.image_size // config.vision_config.patch_size)
                ** 2
            ) // (config.scale_factor**2)
            grid_size = int(tokens_per_tile**0.5)
            if grid_size * grid_size != tokens_per_tile:
                raise ValueError(
                    "image_seq_len must be a perfect square for "
                    "pixel_shuffle_mlp_v2; got "
                    f"{tokens_per_tile} (image_size="
                    f"{config.vision_config.image_size}, patch_size="
                    f"{config.vision_config.patch_size}, scale_factor="
                    f"{config.scale_factor})."
                )
            self.register_buffer(
                "pos_embed_2d",
                _build_2d_sincos_pos_embed(text_hidden_size, grid_size),
                persistent=False,
            )

        # --- DeepStack mergers ---
        self.deepstack_mergers: nn.ModuleList | None = None
        self.use_deepstack = bool(getattr(config, "use_deepstack", False))
        if self.use_deepstack:
            deepstack_visual_indexes = list(
                getattr(config, "deepstack_visual_indexes", []) or []
            )
            self.deepstack_mergers = nn.ModuleList(
                [
                    GraniteDoclingHybridDeepStackMerger(
                        vision_hidden_size,
                        config.scale_factor,
                        text_hidden_size,
                        quant_config=quant_config,
                        prefix=maybe_prefix(
                            prefix, f"deepstack_mergers.{slot}"
                        ),
                    )
                    for slot in range(len(deepstack_visual_indexes))
                ]
            )

    def _apply_main(self, image_hidden_states: torch.Tensor) -> torch.Tensor:
        if self.pooling_mode == "pixel_shuffle":
            x = _pixel_shuffle(image_hidden_states, self.scale_factor)
            return self.modality_projection(x)

        if self.pooling_mode == "pixel_shuffle_mlp":
            x = _pixel_shuffle(image_hidden_states, self.scale_factor)
            x = self.modality_projection(x)
            x = nn.functional.gelu(x)
            x, _ = self.mlp_fc2(x)
            return x

        # pixel_shuffle_mlp_v2
        x = self.ln_in(image_hidden_states)
        x = _pixel_shuffle(x, self.scale_factor)
        x = self.modality_projection(x)
        # ``pos_embed_2d`` is a non-persistent buffer computed deterministically
        # in __init__. vLLM may instantiate the module on ``meta`` first and
        # then ``.to(device)`` it; either way, force device+dtype alignment
        # here so we never trip the cross-device add.
        pos_embed = self.pos_embed_2d
        if pos_embed.device != x.device or pos_embed.is_meta:
            pos_embed = _build_2d_sincos_pos_embed(
                pos_embed.shape[-1], int(pos_embed.shape[0] ** 0.5)
            ).to(device=x.device)
            self.pos_embed_2d = pos_embed
        x = x + pos_embed.to(dtype=x.dtype)
        x = self.ln_mid(x)
        x = nn.functional.gelu(x)
        x, _ = self.mlp_fc2(x)
        x = self.ln_out(x)
        return x

    def forward(
        self,
        image_hidden_states: torch.Tensor,
        deepstack_intermediates: Sequence[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Project vision features into LM space.

        Returns ``[B, image_seq_len, lm_dim]`` in the base case. With DeepStack
        the deepstack tap outputs are projected and concatenated along the last
        dim: ``[B, image_seq_len, lm_dim * (1 + num_levels)]``. This keeps the
        DeepStack residuals attached to the visual embeddings through vLLM's
        per-image embedding pipeline and lets them be split off at the merge
        step.
        """
        main = self._apply_main(image_hidden_states)

        if (
            not self.use_deepstack
            or deepstack_intermediates is None
            or self.deepstack_mergers is None
        ):
            return main

        ds_outputs = [
            self.deepstack_mergers[slot](feat)
            for slot, feat in enumerate(deepstack_intermediates)
        ]
        return torch.cat([main, *ds_outputs], dim=-1)


# -----------------------------------------------------------------------------
# Text model — GraniteMoeHybrid with DeepStack residual injection
# -----------------------------------------------------------------------------


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": 0,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
        "deepstack_input_embeds": 0,
    }
)
class GraniteDoclingHybridTextModel(GraniteMoeHybridModel):
    """``GraniteMoeHybridModel`` with optional per-layer DeepStack injection.

    During prefill the connector packs DeepStack tap outputs into a buffer
    keyed by slot index. After each LM decoder layer whose index appears in
    ``config.deepstack_attn_layers`` we add the corresponding slot's tensor
    to the LM hidden states. The injection is at image-token positions only:
    the buffer is zero for text positions because the scatter in
    ``embed_input_ids`` only writes to ``is_multimodal`` slots.

    DeepStack is restricted to prefill. During autoregressive decoding
    ``deepstack_input_embeds`` is ``None`` and the layer loop is identical
    to the base ``GraniteMoeHybridModel``.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        # ``deepstack_attn_layers`` lives on the (parent) GraniteDoclingHybrid
        # config, but this text model is constructed with the GraniteMoeHybrid
        # text config only — the parent model wires the layer mapping in via
        # `set_deepstack_attn_layers` right after construction.
        self.deepstack_attn_layers: list[int] = []
        self._deepstack_layer_to_slot: dict[int, int] = {}

    def set_deepstack_attn_layers(self, layers: Sequence[int]) -> None:
        self.deepstack_attn_layers = list(layers)
        # Layer-index -> slot lookup for O(1) test inside the layer loop.
        self._deepstack_layer_to_slot = {
            layer_idx: slot for slot, layer_idx in enumerate(self.deepstack_attn_layers)
        }

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        deepstack_input_embeds: IntermediateTensors | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            hidden_states = hidden_states * self.embedding_multiplier
            residual = None
        else:
            if intermediate_tensors is None:
                raise RuntimeError("Intermediate tensors may not be None!")
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for i, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )

            # DeepStack residual injection. The buffer is zero at text-token
            # positions (the scatter in ``embed_input_ids`` only writes to
            # ``is_multimodal`` slots), so the add is a no-op outside images.
            if deepstack_input_embeds is not None:
                slot = self._deepstack_layer_to_slot.get(i)
                if slot is not None:
                    hidden_states = (
                        hidden_states
                        + deepstack_input_embeds[f"deepstack_input_embeds_{slot}"]
                    )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states


# -----------------------------------------------------------------------------
# Composite model: vision + connector + text
# -----------------------------------------------------------------------------


class GraniteDoclingHybridModel(nn.Module):
    """Encapsulates ``vision_model``, ``connector`` and ``text_model``.

    The structure mirrors the HF checkpoint layout
    (``model.{vision_model,connector,text_model}.*``), so weights load via the
    default vLLM auto-loader without any prefix remapping.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config = config

        self.use_deepstack = bool(getattr(config, "use_deepstack", False))
        self.deepstack_visual_indexes: list[int] = list(
            getattr(config, "deepstack_visual_indexes", []) or []
        )

        self.vision_model = GraniteDoclingHybridVisionTransformer(
            config.vision_config,
            quant_config=quant_config,
            deepstack_visual_indexes=(
                self.deepstack_visual_indexes if self.use_deepstack else None
            ),
            prefix=maybe_prefix(prefix, "vision_model"),
        )

        self.connector = GraniteDoclingHybridConnector(
            config,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "connector"),
        )

        self.text_model = GraniteDoclingHybridTextModel(
            vllm_config=vllm_config.with_hf_config(config.text_config),
            prefix=maybe_prefix(prefix, "text_model"),
        )
        if self.use_deepstack:
            self.text_model.set_deepstack_attn_layers(
                getattr(config, "deepstack_attn_layers", []) or []
            )

        self.image_token_id = self.config.image_token_id
        self.image_seq_len = int(
            (config.vision_config.image_size // config.vision_config.patch_size)
            ** 2
            // (config.scale_factor**2)
        )

        # Normalization constants for the uint8 pixel path, created once at
        # init (under the loader's device context) instead of per forward.
        image_mean = getattr(config.vision_config, "image_mean", None)
        image_std = getattr(config.vision_config, "image_std", None)
        self._image_norm_mean: torch.Tensor | None = None
        self._image_norm_std: torch.Tensor | None = None
        if image_mean is not None and image_std is not None:
            self._image_norm_mean = torch.tensor(image_mean).view(1, 3, 1, 1)
            self._image_norm_std = torch.tensor(image_std).view(1, 3, 1, 1)

    # --- helpers ---

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.text_model.embed_input_ids(input_ids)

    def image_pixels_to_features(
        self, pixel_values: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run the vision encoder, returning ``(last_hidden, intermediates)``.

        Handles two input-pixel paths:

        * ``uint8`` — pixels are transferred from the CPU as 1 byte/pixel
          (4× less PCIe bandwidth than fp32) and are rescaled + normalized
          on GPU here (Recipe ⑧ of Huang et al. 2026).
        * float — already-normalized pixel values from the HF processor.

        The processing layer emits a flat, unpadded ``[total_tiles, C, H, W]``
        tensor and every GotOcr2 tile is fully valid, so no padded-tile
        filtering or attention masking is needed here.
        """
        vision_dtype = self.vision_model.embeddings.patch_embedding.weight.dtype

        if pixel_values.dtype == torch.uint8:
            pixel_values = pixel_values.to(dtype=vision_dtype) / 255.0
            if self._image_norm_mean is not None:
                mean = self._image_norm_mean.to(
                    dtype=vision_dtype, device=pixel_values.device
                )
                std = self._image_norm_std.to(
                    dtype=vision_dtype, device=pixel_values.device
                )
                pixel_values = (pixel_values - mean) / std
        else:
            pixel_values = pixel_values.to(dtype=vision_dtype)

        return self.vision_model(pixel_values=pixel_values, patch_attention_mask=None)


# -----------------------------------------------------------------------------
# Top-level model: GraniteDoclingHybridForConditionalGeneration
# -----------------------------------------------------------------------------


@MULTIMODAL_REGISTRY.register_processor(
    GraniteDoclingHybridMultiModalProcessor,
    info=GraniteDoclingHybridProcessingInfo,
    dummy_inputs=GraniteDoclingHybridDummyInputsBuilder,
)
class GraniteDoclingHybridForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsLoRA,
    SupportsPP,
    HasInnerState,
    IsHybrid,
    SupportsMambaPrefixCaching,
):
    """Document-understanding VLM combining a SigLIP-style vision tower, a
    pixel-shuffle modality connector with DeepStack mergers, and a
    GraniteMoeHybrid (Mamba+Attention) text decoder.
    """

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<image>"
        raise ValueError("Only image modality is supported")

    # ---- Mamba state plumbing (delegated to the inner text model) ----

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig):
        text_config = vllm_config.model_config.hf_config.text_config
        return GraniteMoeHybridForCausalLM.get_mamba_state_shape_from_config(
            vllm_config.with_hf_config(text_config)
        )

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig):
        text_config = vllm_config.model_config.hf_config.text_config
        return GraniteMoeHybridForCausalLM.get_mamba_state_dtype_from_config(
            vllm_config.with_hf_config(text_config)
        )

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return GraniteMoeHybridForCausalLM.get_mamba_state_copy_func()

    # ---- construction ----

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config

        self.image_token_id = config.image_token_id
        self.use_deepstack = bool(getattr(config, "use_deepstack", False))
        self.deepstack_visual_indexes: list[int] = list(
            getattr(config, "deepstack_visual_indexes", []) or []
        )
        self.deepstack_attn_layers: list[int] = list(
            getattr(config, "deepstack_attn_layers", []) or []
        )
        if self.use_deepstack:
            if len(self.deepstack_visual_indexes) != len(
                self.deepstack_attn_layers
            ):
                raise ValueError(
                    "deepstack_visual_indexes and deepstack_attn_layers must "
                    "have the same length (got "
                    f"{len(self.deepstack_visual_indexes)} vs "
                    f"{len(self.deepstack_attn_layers)})."
                )
        self.deepstack_num_level = (
            len(self.deepstack_visual_indexes) if self.use_deepstack else 0
        )
        self.visual_dim = config.text_config.hidden_size
        self.multiscale_dim = self.visual_dim * self.deepstack_num_level

        with self._mark_composite_model(
            vllm_config,
            language_targets=(GraniteMoeHybridModel,),
            tower_targets={
                "image": (
                    GraniteDoclingHybridVisionTransformer,
                    GraniteDoclingHybridConnector,
                ),
            },
        ):
            self.model = GraniteDoclingHybridModel(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "model"),
            )

            # DeepStack scratch buffers — one per tap, allocated eagerly (under
            # the loader's device/dtype context) so the LM forward signature is
            # stable from the first warmup run onwards (same as Qwen3-VL).
            if self.use_deepstack:
                self.deepstack_input_embeds = [
                    torch.zeros(
                        vllm_config.scheduler_config.max_num_batched_tokens,
                        self.visual_dim,
                    )
                    for _ in range(self.deepstack_num_level)
                ]
                # Tracks the valid token span currently stored in the buffer.
                # Zero means there is no active deepstack payload to clear.
                self.deepstack_input_embeds_num_tokens = 0

        self.lm_head = ParallelLMHead(
            config.text_config.vocab_size,
            config.text_config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if getattr(config.text_config, "tie_word_embeddings", False):
            self.lm_head.weight = self.model.text_model.embed_tokens.weight

        if hasattr(self.model.text_model, "make_empty_intermediate_tensors"):
            self.make_empty_intermediate_tensors = (
                self.model.text_model.make_empty_intermediate_tensors
            )

        logits_scaling = getattr(config.text_config, "logits_scaling", 1.0)
        self.logits_processor = LogitsProcessor(
            config.text_config.vocab_size,
            scale=1.0 / logits_scaling,
        )

    # ---- input parsing ----

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> ImageInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)

        if pixel_values is None and image_embeds is None:
            return None

        if image_embeds is not None:
            return GraniteDoclingHybridImageEmbeddingInputs(
                type="image_embeds",
                data=image_embeds,
            )

        # Legacy caches may still carry an all-ones mask; it is unused.
        kwargs.pop("pixel_attention_mask", None)
        num_patches = kwargs.pop("num_patches")

        # Use the actual pixel_values shape, not the config — granite-docling's
        # GotOcr2 image processor uses ``size: {height, width}`` rather than
        # ``longest_edge`` and the runtime shape is authoritative.
        if pixel_values.ndim >= 3:
            expected_h = pixel_values.shape[-2]
            expected_w = pixel_values.shape[-1]
        else:
            expected_h = expected_w = self.config.vision_config.image_size

        return GraniteDoclingHybridImagePixelInputs(
            type="pixel_values",
            pixel_values=pixel_values,
            num_patches=num_patches,
            resolve_bindings={"h": expected_h, "w": expected_w},
        )

    # ---- multimodal embedding ----

    def _process_image_input(
        self, image_input: ImageInputs
    ) -> list[torch.Tensor]:
        if image_input["type"] == "image_embeds":
            return list(image_input["data"])

        pixel_values = image_input["pixel_values"]

        last_hidden, intermediate = self.model.image_pixels_to_features(pixel_values)

        ds_intermediates = intermediate if self.use_deepstack else None
        image_features = self.model.connector(
            last_hidden, deepstack_intermediates=ds_intermediates
        )

        num_patches = image_input["num_patches"]
        # ``image_features`` is concatenated along the tile axis 0
        # (shape ``[total_tiles, image_seq_len, lm_dim * (1+L)]``); split per
        # input image and flatten the (tile, token) axes.
        return [
            e.flatten(0, 1)
            for e in image_features.split(num_patches.tolist())
        ]

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return []
        return self._process_image_input(image_input)

    # ---- DeepStack scratch buffer (mirrors Qwen3-VL) ----

    def _get_deepstack_input_embeds(
        self, num_tokens: int
    ) -> IntermediateTensors | None:
        if not getattr(self, "deepstack_input_embeds", None):
            return None
        if num_tokens > self.deepstack_input_embeds[0].size(0):
            self._resize_deepstack_input_embeds(num_tokens)

        return IntermediateTensors(
            {
                f"deepstack_input_embeds_{idx}": self.deepstack_input_embeds[idx][
                    :num_tokens
                ]
                for idx in range(self.deepstack_num_level)
            }
        )

    def _resize_deepstack_input_embeds(self, num_tokens: int) -> None:
        self.deepstack_input_embeds = [
            torch.zeros(
                num_tokens,
                self.visual_dim,
                device=self.deepstack_input_embeds[0].device,
                dtype=self.deepstack_input_embeds[0].dtype,
            )
            for _ in range(self.deepstack_num_level)
        ]

    def _set_deepstack_input_embeds(
        self, deepstack_input_embeds: torch.Tensor
    ) -> None:
        """Copy per-level multiscale embeddings into the scratch buffer.

        ``deepstack_input_embeds`` has shape ``[num_levels, num_tokens, dim]``.
        """
        if not getattr(self, "deepstack_input_embeds", None):
            return
        num_tokens = deepstack_input_embeds.size(1)
        if num_tokens > self.deepstack_input_embeds[0].size(0):
            self._resize_deepstack_input_embeds(num_tokens)
        for idx in range(self.deepstack_num_level):
            self.deepstack_input_embeds[idx][:num_tokens].copy_(
                deepstack_input_embeds[idx]
            )
        self.deepstack_input_embeds_num_tokens = num_tokens

    def _clear_deepstack_input_embeds(self, num_tokens: int) -> None:
        if not getattr(self, "deepstack_input_embeds", None):
            return
        # Skip the zeroing kernels on steps that never staged a payload
        # (i.e. every decode-only step).
        if getattr(self, "deepstack_input_embeds_num_tokens", 0) == 0:
            return
        if num_tokens > 0:
            for idx in range(self.deepstack_num_level):
                self.deepstack_input_embeds[idx][:num_tokens].zero_()
            self.deepstack_input_embeds_num_tokens = 0

    def _compute_deepstack_embeds(
        self,
        inputs_embeds: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings,
        is_multimodal: torch.Tensor,
    ) -> tuple[torch.Tensor, MultiModalEmbeddings]:
        """Split DeepStack channels off the multimodal embeddings.

        The connector concatenates main + per-tap features along the hidden
        dim. Here we split them back, project the multiscale portion into a
        ``[seq_len, num_levels * visual_dim]`` tensor scattered at image-token
        positions, and reshape into a ``[num_levels, seq_len, visual_dim]``
        view that the LM consumes one slot at a time.
        """
        visual_lens = [len(x) for x in multimodal_embeddings]
        cat = torch.cat(list(multimodal_embeddings), dim=0)
        main, multiscale = torch.split(
            cat,
            [self.visual_dim, self.multiscale_dim],
            dim=-1,
        )

        multimodal_embeddings_main = main.split(visual_lens, dim=0)
        multimodal_embeddings_multiscale = multiscale.split(visual_lens, dim=0)

        deepstack_input_embeds = inputs_embeds.new_zeros(
            inputs_embeds.size(0),
            self.deepstack_num_level * inputs_embeds.size(1),
        )
        deepstack_input_embeds = _merge_multimodal_embeddings(
            inputs_embeds=deepstack_input_embeds,
            multimodal_embeddings=multimodal_embeddings_multiscale,
            is_multimodal=is_multimodal,
        )
        deepstack_input_embeds = deepstack_input_embeds.view(
            inputs_embeds.size(0), self.deepstack_num_level, self.visual_dim
        ).permute(1, 0, 2)

        return deepstack_input_embeds, multimodal_embeddings_main

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self._embed_text_input_ids(
            input_ids,
            self.model.embed_input_ids,
            is_multimodal=is_multimodal,
        )

        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds

        is_multimodal = _require_is_multimodal(is_multimodal)

        if self.use_deepstack:
            deepstack_input_embeds, multimodal_embeddings = (
                self._compute_deepstack_embeds(
                    inputs_embeds=inputs_embeds,
                    multimodal_embeddings=multimodal_embeddings,
                    is_multimodal=is_multimodal,
                )
            )
        else:
            deepstack_input_embeds = None

        inputs_embeds = _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

        if deepstack_input_embeds is not None:
            self._set_deepstack_input_embeds(deepstack_input_embeds)

        return inputs_embeds

    # ---- forward ----

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None

        deepstack_input_embeds: IntermediateTensors | None = None
        if (
            self.use_deepstack
            and inputs_embeds is not None
            and get_pp_group().is_first_rank
        ):
            deepstack_input_embeds = self._get_deepstack_input_embeds(
                inputs_embeds.size(0)
            )

        hidden_states = self.model.text_model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds=inputs_embeds,
            deepstack_input_embeds=deepstack_input_embeds,
        )

        # Clear the scratch buffer so any subsequent prefill in the same
        # process starts from zero positions (the merge logic relies on
        # text-position entries being zero).
        if (
            self.use_deepstack
            and inputs_embeds is not None
            and get_pp_group().is_first_rank
        ):
            self._clear_deepstack_input_embeds(inputs_embeds.size(0))

        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states)

    # ---- weight loading ----

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)

    # ---- LoRA / module mapping ----

    def get_mm_mapping(self) -> MultiModelKeys:
        return MultiModelKeys.from_string_field(
            language_model="model.text_model",
            connector="model.connector",
            tower_model="model.vision_model",
        )

    def get_num_mm_encoder_tokens(self, num_image_tokens: int) -> int:
        return num_image_tokens * self.config.scale_factor**2

    def get_num_mm_connector_tokens(self, num_vision_tokens: int) -> int:
        return num_vision_tokens // self.config.scale_factor**2
