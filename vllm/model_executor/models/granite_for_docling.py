# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only GraniteForDocling VLM compatible with HuggingFace weights.

* Tiled vision encoder with DeepStack intermediate taps.
* Pixel-shuffle modality projector with a coarse and a fine (4x tokens) path.
* Dense Granite text decoder with a gated MLP and DeepStack injection.
* Optional multi-token prediction heads for speculative decoding.
"""

from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, Literal, TypeAlias

import torch
from torch import nn
from transformers import BatchFeature, GraniteForDoclingProcessor
from transformers.models.granite_for_docling.image_processing_granite_for_docling import (  # noqa: E501
    get_all_supported_aspect_ratios,
)

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.distributed.parallel_state import get_pp_group
from vllm.inputs import MultiModalDataDict
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import Attention, RSWAAttention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
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

from .granite_for_docling_mtp import maybe_build_mtp
from .idefics2_vision_model import (
    Idefics2VisionTransformer as _BaseIdefics3VisionTransformer,
)
from .interfaces import (
    MultiModalEmbeddings,
    SupportsLoRA,
    SupportsMultiModal,
    SupportsPP,
    _require_is_multimodal,
)
from .utils import (
    AutoWeightsLoader,
    _merge_multimodal_embeddings,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)


# -----------------------------------------------------------------------------
# Tensor schemas
# -----------------------------------------------------------------------------


class GraniteForDoclingImagePixelInputs(TensorSchema):
    """
    Dimensions:
        - bnp: Batch size * number of images * number of tiles
        - c: Number of channels (3)
        - h: Height
        - w: Width
        - bn: Batch size * number of images
    """

    type: Literal["pixel_values"]
    pixel_values: Annotated[torch.Tensor, TensorShape("bnp", 3, "h", "w")]
    num_patches: Annotated[torch.Tensor, TensorShape("bn")]
    tile_fine_mask: Annotated[torch.Tensor, TensorShape("bnp")]


class GraniteForDoclingImageEmbeddingInputs(TensorSchema):
    """
    Dimensions:
        - bn: Batch size * number of images
        - f: Image feature size
        - h: Hidden size of the language model * ``1 + num_deepstack_levels``
    """

    type: Literal["image_embeds"]
    data: Annotated[torch.Tensor, TensorShape("bn", "f", "h")]


ImageInputs: TypeAlias = (
    GraniteForDoclingImagePixelInputs | GraniteForDoclingImageEmbeddingInputs
)


# -----------------------------------------------------------------------------
# Processing
# -----------------------------------------------------------------------------


class GraniteForDoclingProcessingInfo(BaseProcessingInfo):
    def get_hf_processor(
        self, *, fine_route: bool | None = None, **kwargs: object
    ) -> GraniteForDoclingProcessor:
        # ``fine_route`` is a per-call option of the processor, not an init argument.
        return self.ctx.get_hf_processor(GraniteForDoclingProcessor, **kwargs)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}

    def get_image_kwargs(self, mm_kwargs: Mapping[str, object]) -> dict[str, object]:
        """Per-request kwargs merged with the model's ``mm_processor_kwargs``."""
        return self.ctx.get_merged_mm_kwargs(mm_kwargs)

    def get_image_seq_len(
        self,
        processor: GraniteForDoclingProcessor,
        mm_kwargs: Mapping[str, object],
    ) -> int:
        image_seq_len = processor.image_seq_len
        if self.get_image_kwargs(mm_kwargs).get("fine_route", False):
            return image_seq_len * 4
        return image_seq_len

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        processor: GraniteForDoclingProcessor,
        mm_kwargs: Mapping[str, object],
    ) -> int:
        num_patches, _, _ = processor.image_processor.get_number_of_image_patches(
            image_height, image_width, self.get_image_kwargs(mm_kwargs)
        )
        return num_patches * self.get_image_seq_len(processor, mm_kwargs)

    def get_image_repl(
        self,
        *,
        image_width: int,
        image_height: int,
        processor: GraniteForDoclingProcessor,
        mm_kwargs: Mapping[str, object],
    ) -> str:
        image_kwargs = self.get_image_kwargs(mm_kwargs)
        _, num_rows, num_cols = processor.image_processor.get_number_of_image_patches(
            image_height, image_width, image_kwargs
        )
        return processor.replace_image_token(
            {"rows": [[num_rows]], "cols": [[num_cols]]},
            0,
            fine_route=bool(image_kwargs.get("fine_route", False)),
        )


class GraniteForDoclingDummyInputsBuilder(
    BaseDummyInputsBuilder[GraniteForDoclingProcessingInfo]
):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        return self.info.get_hf_processor().image_token * num_images

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        image_processor = self.info.get_hf_processor().image_processor
        # The grid with the most tiles the image processor can select
        num_cols, num_rows = max(
            get_all_supported_aspect_ratios(
                image_processor.min_patches, image_processor.max_patches
            ),
            key=lambda grid: grid[0] * grid[1],
        )
        image_overrides = mm_options.get("image") if mm_options else None

        return {
            "image": self._get_dummy_images(
                width=image_processor.size["width"] * num_cols,
                height=image_processor.size["height"] * num_rows,
                num_images=num_images,
                overrides=image_overrides,
            )
        }


class GraniteForDoclingMultiModalProcessor(
    BaseMultiModalProcessor[GraniteForDoclingProcessingInfo]
):
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        # Text-only input: go straight to the tokenizer
        if not (images := mm_data.get("images", [])):
            prompt_ids = self.info.get_tokenizer().encode(prompt)
            prompt_ids = self._apply_hf_processor_tokens_only(prompt_ids)
            return BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")

        processed_outputs = super()._call_hf_processor(
            prompt, mm_data, mm_kwargs, tok_kwargs
        )

        # One prompt is one sample: ``(1, num_tiles, C, H, W)`` without padding tiles
        pixel_values = processed_outputs["pixel_values"].flatten(0, 1)
        processed_outputs["pixel_values"] = pixel_values
        tile_fine_mask = processed_outputs.get("tile_fine_mask")
        processed_outputs["tile_fine_mask"] = (
            tile_fine_mask.flatten()
            if tile_fine_mask is not None
            else torch.zeros(pixel_values.shape[0], dtype=torch.bool)
        )

        image_processor = self.info.get_hf_processor(**mm_kwargs).image_processor
        image_kwargs = self.info.get_image_kwargs(mm_kwargs)
        parsed_images = self.info.parse_mm_data(
            {"image": images}, validate=False
        ).get_items("image", ImageProcessorItems)
        processed_outputs["num_patches"] = torch.tensor(
            [
                image_processor.get_number_of_image_patches(
                    image_size.height, image_size.width, image_kwargs
                )[0]
                for image_size in map(
                    parsed_images.get_image_size, range(len(parsed_images))
                )
            ]
        )
        return processed_outputs

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        num_patches = hf_inputs.get("num_patches", torch.empty(0))

        return dict(
            pixel_values=MultiModalFieldConfig.flat_from_sizes("image", num_patches),
            tile_fine_mask=MultiModalFieldConfig.flat_from_sizes("image", num_patches),
            num_patches=MultiModalFieldConfig.batched("image"),
            image_embeds=MultiModalFieldConfig.batched("image"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        image_token = hf_processor.image_token

        def get_replacement(item_idx: int) -> PromptUpdateDetails:
            images = mm_items.get_items("image", ImageProcessorItems)
            image_size = images.get_image_size(item_idx)
            image_repl = self.info.get_image_repl(
                image_width=image_size.width,
                image_height=image_size.height,
                processor=hf_processor,
                mm_kwargs=hf_processor_mm_kwargs,
            )
            return PromptUpdateDetails.select_text(image_repl, embed_text=image_token)

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


class GraniteForDoclingVisionTransformer(_BaseIdefics3VisionTransformer):
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
        # Cache for the fast full-tile position-ids path (see _fast_position_ids).
        self._full_position_ids_cache: dict[
            tuple[int, int, torch.device], torch.Tensor
        ] = {}
        # Pin ViT to FA3; decoder R-SWA forces FA4 globally but FA4 fails on SigLIP.
        from vllm.model_executor.layers.attention.mm_encoder_attention import (
            MMEncoderAttention,
        )
        from vllm.v1.attention.backends.fa_utils import is_fa_version_supported

        for module in self.modules():
            if (
                isinstance(module, MMEncoderAttention)
                and getattr(module, "_fa_version", None) == 4
                and is_fa_version_supported(3)
            ):
                module._fa_version = 3

    def _fast_position_ids(
        self, num_patches_h: int, num_patches_w: int, device: torch.device
    ) -> torch.Tensor:
        """Position ids for a fully-valid (all-ones patch_attention_mask) tile grid.

        ``Idefics2VisionEmbeddings.forward`` computes this per-tile inside a Python
        loop via ``.sum()``/``.item()``/``nonzero()`` on the mask — a GPU->CPU sync
        per tile. GotOcr2 tiles are always the same fixed size and fully valid
        (see the all-ones mask synthesized below), so the result is identical
        across every tile and call; compute it once per grid shape instead.
        """
        key = (num_patches_h, num_patches_w, device)
        cached = self._full_position_ids_cache.get(key)
        if cached is not None:
            return cached
        embeddings_module = self.embeddings
        num_patches_per_side = embeddings_module.num_patches_per_side
        boundaries = torch.arange(
            1 / num_patches_per_side, 1.0, 1 / num_patches_per_side, device=device
        )
        fractional_coords_h = torch.arange(
            0, 1 - 1e-6, 1 / num_patches_h, device=device
        )
        fractional_coords_w = torch.arange(
            0, 1 - 1e-6, 1 / num_patches_w, device=device
        )
        bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
        bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)
        pos_ids = (
            bucket_coords_h[:, None] * num_patches_per_side + bucket_coords_w
        ).flatten()
        self._full_position_ids_cache[key] = pos_ids
        return pos_ids

    def _fast_embeddings(self, pixel_values: torch.Tensor) -> torch.Tensor:
        embeddings_module = self.embeddings
        batch_size, _, max_im_h, max_im_w = pixel_values.shape
        target_dtype = embeddings_module.patch_embedding.weight.dtype
        patch_embeds = embeddings_module.patch_embedding(pixel_values.to(target_dtype))
        embeddings = patch_embeds.flatten(2).transpose(1, 2)
        num_patches_h = max_im_h // embeddings_module.patch_size
        num_patches_w = max_im_w // embeddings_module.patch_size
        pos_ids = self._fast_position_ids(
            num_patches_h, num_patches_w, pixel_values.device
        )
        position_ids = pos_ids.unsqueeze(0).expand(batch_size, -1)
        return embeddings + embeddings_module.position_embedding(position_ids)

    def forward(
        self,
        pixel_values: torch.Tensor,
        patch_attention_mask: torch.BoolTensor | None = None,
        tgt_sizes: torch.IntTensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if patch_attention_mask is None and tgt_sizes is None:
            # GotOcr2 tiles are always fully valid and the same fixed size, so
            # skip Idefics2VisionEmbeddings' generic per-tile CPU-sync loop.
            hidden_states = self._fast_embeddings(pixel_values)
        else:
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
# Connector (modality projector)
# -----------------------------------------------------------------------------


def _build_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
    """2D sine-cosine position embeddings of shape ``[grid_size**2, embed_dim]``
    for a square grid of tokens in row-major order."""
    omega = torch.arange(embed_dim // 4, dtype=torch.float32) / (embed_dim // 4)
    omega = 1.0 / 10000**omega
    positions = torch.arange(grid_size, dtype=torch.float32)
    rows, cols = torch.meshgrid(positions, positions, indexing="ij")
    out_cols = cols.reshape(-1, 1) * omega
    out_rows = rows.reshape(-1, 1) * omega
    return torch.cat(
        [out_cols.sin(), out_cols.cos(), out_rows.sin(), out_rows.cos()], dim=1
    )


def _pixel_shuffle(x: torch.Tensor, scale_factor: int) -> torch.Tensor:
    """Spatial-to-channel pixel shuffle."""
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
    x = x.reshape(bsz, seq // (scale_factor**2), embed_dim * (scale_factor**2))
    return x


class GraniteForDoclingDeepStackMerger(nn.Module):
    """Projects pixel-shuffled intermediate vision features to the LM hidden size."""

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


class GraniteForDoclingSimpleMLP(nn.Module):
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


class GraniteForDoclingConnector(nn.Module):
    """Pixel-shuffle modality projector with a coarse and a fine path.

    The fine path shuffles by half the factor and yields four times as many
    image tokens per tile. Both paths share ``ln_in``, ``ln_mid``, ``mlp_fc2``
    and ``ln_out``; each has its own projection, position embedding and
    DeepStack mergers.
    """

    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        text_hidden_size = config.text_config.hidden_size
        vision_hidden_size = config.vision_config.hidden_size
        grid_size = config.vision_config.image_size // config.vision_config.patch_size
        self.scale_factor = config.scale_factor
        self.fine_scale_factor = config.scale_factor // 2

        self.ln_in = nn.LayerNorm(vision_hidden_size)
        self.modality_projection = GraniteForDoclingSimpleMLP(
            config,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "modality_projection"),
        )
        self.proj_fine = ReplicatedLinear(
            vision_hidden_size * (self.fine_scale_factor**2),
            text_hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "proj_fine"),
        )
        self.ln_mid = nn.LayerNorm(text_hidden_size)
        self.mlp_fc2 = ReplicatedLinear(
            text_hidden_size,
            text_hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "mlp_fc2"),
        )
        self.ln_out = nn.LayerNorm(text_hidden_size)
        self.register_buffer(
            "pos_embed_2d",
            _build_2d_sincos_pos_embed(
                text_hidden_size, grid_size // self.scale_factor
            ),
            persistent=False,
        )
        self.register_buffer(
            "pos_embed_2d_fine",
            _build_2d_sincos_pos_embed(
                text_hidden_size, grid_size // self.fine_scale_factor
            ),
            persistent=False,
        )
        self.deepstack_mergers = nn.ModuleList(
            [
                GraniteForDoclingDeepStackMerger(
                    vision_hidden_size,
                    self.scale_factor,
                    text_hidden_size,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, f"deepstack_mergers.{slot}"),
                )
                for slot in range(len(config.deepstack_visual_indexes))
            ]
        )
        self.deepstack_mergers_fine = nn.ModuleList(
            [
                GraniteForDoclingDeepStackMerger(
                    vision_hidden_size,
                    self.fine_scale_factor,
                    text_hidden_size,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, f"deepstack_mergers_fine.{slot}"),
                )
                for slot in range(len(config.deepstack_visual_indexes))
            ]
        )

    def _pos_embed(self, name: str, x: torch.Tensor) -> torch.Tensor:
        # Non-persistent buffers are not loaded from the checkpoint; rebuild them
        # if the module was instantiated on the meta device.
        pos_embed = getattr(self, name)
        if pos_embed.is_meta or pos_embed.device != x.device:
            pos_embed = _build_2d_sincos_pos_embed(
                pos_embed.shape[-1], int(pos_embed.shape[0] ** 0.5)
            ).to(device=x.device)
            setattr(self, name, pos_embed)
        return pos_embed.to(dtype=x.dtype)

    def _project(
        self,
        image_hidden_states: torch.Tensor,
        deepstack_intermediates: Sequence[torch.Tensor],
        fine: bool,
    ) -> torch.Tensor:
        x = self.ln_in(image_hidden_states)
        if fine:
            x, _ = self.proj_fine(_pixel_shuffle(x, self.fine_scale_factor))
            x = x + self._pos_embed("pos_embed_2d_fine", x)
            mergers = self.deepstack_mergers_fine
        else:
            x = self.modality_projection(_pixel_shuffle(x, self.scale_factor))
            x = x + self._pos_embed("pos_embed_2d", x)
            mergers = self.deepstack_mergers
        x = nn.functional.gelu(self.ln_mid(x))
        x, _ = self.mlp_fc2(x)
        x = self.ln_out(x)
        return torch.cat(
            [
                x,
                *(
                    merger(feat)
                    for merger, feat in zip(mergers, deepstack_intermediates)
                ),
            ],
            dim=-1,
        )

    def forward(
        self,
        image_hidden_states: torch.Tensor,
        deepstack_intermediates: Sequence[torch.Tensor],
        tile_fine_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Projects every tile to image tokens with the DeepStack features
        concatenated along the last dim, flattened in tile order:
        ``[sum(tokens_per_tile), lm_dim * (1 + num_levels)]``."""
        if not tile_fine_mask.any():
            return self._project(
                image_hidden_states, deepstack_intermediates, fine=False
            ).flatten(0, 1)
        if tile_fine_mask.all():
            return self._project(
                image_hidden_states, deepstack_intermediates, fine=True
            ).flatten(0, 1)

        coarse_mask = ~tile_fine_mask
        coarse = self._project(
            image_hidden_states[coarse_mask],
            [feat[coarse_mask] for feat in deepstack_intermediates],
            fine=False,
        )
        fine = self._project(
            image_hidden_states[tile_fine_mask],
            [feat[tile_fine_mask] for feat in deepstack_intermediates],
            fine=True,
        )
        tiles: list[torch.Tensor | None] = [None] * tile_fine_mask.shape[0]
        for tile_idx, features in zip(
            coarse_mask.nonzero(as_tuple=True)[0].tolist(), coarse
        ):
            tiles[tile_idx] = features
        for tile_idx, features in zip(
            tile_fine_mask.nonzero(as_tuple=True)[0].tolist(), fine
        ):
            tiles[tile_idx] = features
        return torch.cat(tiles, dim=0)


# -----------------------------------------------------------------------------
# Text model — dense attention decoder with DeepStack residual injection
# -----------------------------------------------------------------------------


class GraniteForDoclingSharedMLP(nn.Module):
    """Shared SwiGLU MLP using HF ``shared_mlp.{input_linear,output_linear}`` keys."""

    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.input_linear = MergedColumnParallelLinear(
            input_size=config.hidden_size,
            output_sizes=[config.shared_intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.input_linear",
        )
        self.output_linear = RowParallelLinear(
            config.shared_intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.output_linear",
        )
        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. Only silu is supported."
            )
        self.act_fn = SiluAndMul()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.input_linear(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states, _ = self.output_linear(hidden_states)
        return hidden_states


def resolve_rswa_full_attn_layers(config) -> frozenset[int]:
    """Decoder layers that keep full attention in a hybrid R-SWA stack.

    The full-attention layer set must match the schedule the weights were
    trained with. ``rswa_full_attn_layers`` (explicit index list) wins over
    ``rswa_hybrid_period`` (every N-th layer, N>1).
    """
    num_layers = int(getattr(config, "num_hidden_layers", 0) or 0)
    if num_layers <= 0:
        return frozenset()
    explicit = getattr(config, "rswa_full_attn_layers", None)
    if explicit:
        return frozenset(int(i) % num_layers for i in explicit)
    period = int(getattr(config, "rswa_hybrid_period", 0) or 0)
    if period > 1:
        return frozenset(i for i in range(num_layers) if (i + 1) % period == 0)
    return frozenset()


class GraniteForDoclingAttention(nn.Module):
    def __init__(
        self,
        config,
        layer_idx: int = 0,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.attention_bias = getattr(config, "attention_bias", False)
        self.attention_multiplier = getattr(config, "attention_multiplier", 1.0)
        self.total_num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.total_num_heads
        self.total_num_kv_heads = config.num_key_value_heads

        tp_size = get_tensor_model_parallel_world_size()
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_key_value_heads = max(1, self.total_num_kv_heads // tp_size)

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=self.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.hidden_size,
            self.hidden_size,
            bias=self.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=config.rope_parameters,
            is_neox_style=True,
        )

        rswa_window = getattr(config, "rswa_window", None)
        if rswa_window and layer_idx in resolve_rswa_full_attn_layers(config):
            # Hybrid stack: this layer sees the whole sequence.
            rswa_window = None
        if rswa_window:
            self.attn = RSWAAttention(
                self.num_heads,
                self.head_dim,
                self.attention_multiplier,
                num_kv_heads=self.num_key_value_heads,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.attn",
                rswa_window=rswa_window,
            )
        else:
            self.attn = Attention(
                self.num_heads,
                self.head_dim,
                self.attention_multiplier,
                num_kv_heads=self.num_key_value_heads,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.attn",
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        query, key, value = qkv.split(
            [
                self.num_heads * self.head_dim,
                self.num_key_value_heads * self.head_dim,
                self.num_key_value_heads * self.head_dim,
            ],
            dim=-1,
        )
        query, key = self.rotary_emb(positions, query, key)
        hidden_states = self.attn(query, key, value)
        del query, key, value
        hidden_states = self.o_proj(hidden_states)[0]
        return hidden_states


class GraniteForDoclingDecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        layer_idx: int = 0,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.residual_multiplier = getattr(config, "residual_multiplier", 1.0)
        self.self_attn = GraniteForDoclingAttention(
            config,
            layer_idx=layer_idx,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.shared_mlp = GraniteForDoclingSharedMLP(
            config, quant_config=quant_config, prefix=f"{prefix}.shared_mlp"
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states = residual + hidden_states * self.residual_multiplier

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.shared_mlp(hidden_states)
        hidden_states = residual + hidden_states * self.residual_multiplier
        return hidden_states, residual


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": 0,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
        "deepstack_input_embeds": 0,
    }
)
class GraniteForDoclingTextModel(nn.Module):
    """Dense Granite decoder with optional per-layer DeepStack injection."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.quant_config = quant_config
        self.vocab_size = config.vocab_size
        self.embedding_multiplier = getattr(config, "embedding_multiplier", 1.0)

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )

        def get_layer(prefix: str):
            layer_idx = int(prefix.rsplit(".", 1)[1])
            return GraniteForDoclingDecoderLayer(
                config,
                layer_idx=layer_idx,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.deepstack_attn_layers: list[int] = []
        self._deepstack_layer_to_slot: dict[int, int] = {}

    def set_deepstack_attn_layers(self, layers: Sequence[int]) -> None:
        self.deepstack_attn_layers = list(layers)
        self._deepstack_layer_to_slot = {
            layer_idx: slot for slot, layer_idx in enumerate(self.deepstack_attn_layers)
        }

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

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

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        from vllm.model_executor.model_loader.weight_utils import default_weight_loader

        from .utils import is_pp_missing_parameter

        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            if is_pp_missing_parameter(name, self):
                continue
            mapped = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name in name:
                    mapped_name = name.replace(weight_name, param_name)
                    if mapped_name not in params_dict:
                        continue
                    param = params_dict[mapped_name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight, shard_id)
                    loaded_params.add(mapped_name)
                    mapped = True
                    break
            if mapped:
                continue
            if name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


# -----------------------------------------------------------------------------
# Composite model: vision + connector + text
# -----------------------------------------------------------------------------


class GraniteForDoclingModel(nn.Module):
    """``vision_model``, ``connector`` and ``text_model``, laid out like the HF
    checkpoint (``model.{vision_model,connector,text_model}.*``)."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config

        self.vision_model = GraniteForDoclingVisionTransformer(
            config.vision_config,
            quant_config=quant_config,
            deepstack_visual_indexes=config.deepstack_visual_indexes,
            prefix=maybe_prefix(prefix, "vision_model"),
        )
        self.connector = GraniteForDoclingConnector(
            config,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "connector"),
        )
        self.text_model = GraniteForDoclingTextModel(
            vllm_config=vllm_config.with_hf_config(config.text_config),
            prefix=maybe_prefix(prefix, "text_model"),
        )
        self.text_model.set_deepstack_attn_layers(config.deepstack_attn_layers)
        self.image_seq_len = (
            config.vision_config.image_size // config.vision_config.patch_size
        ) ** 2 // config.scale_factor**2

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.text_model.embed_input_ids(input_ids)

    def image_pixels_to_features(
        self, pixel_values: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        vision_dtype = self.vision_model.embeddings.patch_embedding.weight.dtype
        return self.vision_model(
            pixel_values=pixel_values.to(dtype=vision_dtype),
            patch_attention_mask=None,
        )


# -----------------------------------------------------------------------------
# Top-level model: GraniteForDoclingForConditionalGeneration
# -----------------------------------------------------------------------------


@MULTIMODAL_REGISTRY.register_processor(
    GraniteForDoclingMultiModalProcessor,
    info=GraniteForDoclingProcessingInfo,
    dummy_inputs=GraniteForDoclingDummyInputsBuilder,
)
class GraniteForDoclingForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsLoRA,
    SupportsPP,
):
    """Document conversion VLM: tiled vision tower, pixel-shuffle connector with
    DeepStack injection and a dense Granite text decoder."""

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    }

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<image>"

        raise ValueError("Only image modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config
        self.image_token_id = config.image_token_id
        self.deepstack_num_level = len(config.deepstack_visual_indexes)
        self.visual_dim = config.text_config.hidden_size
        self.multiscale_dim = self.visual_dim * self.deepstack_num_level

        with self._mark_composite_model(
            vllm_config,
            language_targets=(GraniteForDoclingTextModel,),
            tower_targets={
                "image": (
                    GraniteForDoclingVisionTransformer,
                    GraniteForDoclingConnector,
                ),
            },
        ):
            self.model = GraniteForDoclingModel(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "model"),
            )
            # DeepStack scratch buffers, one per tap, allocated eagerly so the
            # LM forward signature is stable from the first warmup run onwards.
            self.deepstack_input_embeds = [
                torch.zeros(
                    vllm_config.scheduler_config.max_num_batched_tokens,
                    self.visual_dim,
                )
                for _ in range(self.deepstack_num_level)
            ]
            # Valid token span currently staged in the buffers; zero means
            # there is nothing to clear.
            self.deepstack_input_embeds_num_tokens = 0

        self.lm_head = ParallelLMHead(
            config.text_config.vocab_size,
            config.text_config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if config.text_config.tie_word_embeddings:
            self.lm_head.weight = self.model.text_model.embed_tokens.weight
        self.make_empty_intermediate_tensors = (
            self.model.text_model.make_empty_intermediate_tensors
        )
        self.logits_processor = LogitsProcessor(
            config.text_config.vocab_size,
            scale=1.0 / config.text_config.logits_scaling,
        )
        self.mtp = maybe_build_mtp(
            config, vllm_config, prefix=maybe_prefix(prefix, "mtp")
        )

    def _parse_and_validate_image_input(self, **kwargs: object) -> ImageInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)

        if pixel_values is None and image_embeds is None:
            return None

        if image_embeds is not None:
            return GraniteForDoclingImageEmbeddingInputs(
                type="image_embeds",
                data=image_embeds,
            )

        image_size = self.config.vision_config.image_size
        return GraniteForDoclingImagePixelInputs(
            type="pixel_values",
            pixel_values=pixel_values,
            num_patches=kwargs.pop("num_patches"),
            tile_fine_mask=kwargs.pop("tile_fine_mask"),
            resolve_bindings={"h": image_size, "w": image_size},
        )

    def _process_image_input(self, image_input: ImageInputs) -> list[torch.Tensor]:
        if image_input["type"] == "image_embeds":
            return list(image_input["data"])

        tile_fine_mask = image_input["tile_fine_mask"]
        last_hidden, intermediates = self.model.image_pixels_to_features(
            image_input["pixel_values"]
        )
        image_features = self.model.connector(
            last_hidden, intermediates, tile_fine_mask
        )

        # Split the flat tile-major features per image: coarse tiles hold
        # ``image_seq_len`` tokens, fine tiles four times as many.
        num_patches = image_input["num_patches"].tolist()
        num_fine = torch.stack(
            [mask.sum() for mask in tile_fine_mask.split(num_patches)]
        ).tolist()
        image_seq_len = self.model.image_seq_len
        tokens_per_image = [
            image_seq_len * (n + 3 * f) for n, f in zip(num_patches, num_fine)
        ]
        return list(image_features.split(tokens_per_image))

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return []

        return self._process_image_input(image_input)

    # ---- DeepStack scratch buffer (mirrors Qwen3-VL) ----

    def _get_deepstack_input_embeds(self, num_tokens: int) -> IntermediateTensors:
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

    def _set_deepstack_input_embeds(self, deepstack_input_embeds: torch.Tensor) -> None:
        """Copy ``[num_levels, num_tokens, dim]`` embeddings into the buffers."""
        num_tokens = deepstack_input_embeds.size(1)
        if num_tokens > self.deepstack_input_embeds[0].size(0):
            self._resize_deepstack_input_embeds(num_tokens)
        for idx in range(self.deepstack_num_level):
            self.deepstack_input_embeds[idx][:num_tokens].copy_(
                deepstack_input_embeds[idx]
            )
        self.deepstack_input_embeds_num_tokens = num_tokens

    def _clear_deepstack_input_embeds(self, num_tokens: int) -> None:
        # Decode-only steps never stage a payload, skip the zeroing kernels.
        if self.deepstack_input_embeds_num_tokens == 0:
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
        """Split the DeepStack channels off the multimodal embeddings.

        The connector concatenates main + per-tap features along the hidden
        dim. The multiscale part is scattered at image-token positions into a
        ``[num_levels, seq_len, visual_dim]`` tensor the LM consumes one slot
        at a time.
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
        deepstack_input_embeds, multimodal_embeddings = self._compute_deepstack_embeds(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )
        inputs_embeds = _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )
        self._set_deepstack_input_embeds(deepstack_input_embeds)
        return inputs_embeds

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
        if inputs_embeds is not None and get_pp_group().is_first_rank:
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

        # The merge relies on text positions being zero in the scratch buffers.
        if inputs_embeds is not None and get_pp_group().is_first_rank:
            self._clear_deepstack_input_embeds(inputs_embeds.size(0))

        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # The density router only serves the two-pass routing API of the HF model.
        loader = AutoWeightsLoader(self, skip_prefixes=["model.density_router."])
        return loader.load_weights(weights)

    def get_language_model(self) -> nn.Module:
        return self.model.text_model

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
