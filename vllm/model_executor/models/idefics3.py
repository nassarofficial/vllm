# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2024 the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Idefics3 model compatible with HuggingFace weights."""

import copy
from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, Literal, TypeAlias

import torch
from torch import nn
from transformers import (
    BatchFeature,
    Idefics3Config,
    Idefics3ImageProcessor,
    Idefics3Processor,
)

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateCopyFunc
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

from .idefics2_vision_model import (
    Idefics2VisionTransformer as Idefics3VisionTransformer,
)
from .interfaces import (
    HasInnerState,
    IsHybrid,
    MultiModalEmbeddings,
    SupportsLoRA,
    SupportsMambaPrefixCaching,
    SupportsMultiModal,
    SupportsPP,
)
from .llama import LlamaModel
from .granitemoehybrid import GraniteMoeHybridForCausalLM, GraniteMoeHybridModel
from .utils import AutoWeightsLoader, maybe_prefix

logger = init_logger(__name__)


class Idefics3ImagePixelInputs(TensorSchema):
    """
    Dimensions:
        - bn: Batch size * number of images
        - bnp: Batch size * number of images * number of patches
        - c: Number of channels (3)
        - h: Height
        - w: Width
    """

    type: Literal["pixel_values"]
    pixel_values: Annotated[torch.Tensor, TensorShape("bnp", 3, "h", "w")]
    pixel_attention_mask: Annotated[torch.Tensor, TensorShape("bnp", "h", "w")]
    num_patches: Annotated[torch.Tensor, TensorShape("bn")]


class Idefics3ImageEmbeddingInputs(TensorSchema):
    """
    Dimensions:
        - bn: Batch size * number of images
        - f: Image feature size
        - h: Hidden size (must match the hidden size of language model backbone)
    """

    type: Literal["image_embeds"]
    data: Annotated[torch.Tensor, TensorShape("bn", "f", "h")]


ImageInputs: TypeAlias = Idefics3ImagePixelInputs | Idefics3ImageEmbeddingInputs


class Idefics3ProcessingInfo(BaseProcessingInfo):
    def get_hf_processor(self, **kwargs: object) -> Idefics3Processor:
        processor = self.ctx.get_hf_processor(Idefics3Processor, **kwargs)
        # When the model specifies GotOcr2ImageProcessor (crop_to_patches
        # mode) in its preprocessor config, the Idefics3Processor may load
        # an Idefics3ImageProcessor instead.  The Idefics3ImageProcessor uses
        # its own default max_image_size (364) producing patches that are
        # incompatible with the vision model.  Swap in the correct processor.
        ip = processor.image_processor
        if (getattr(ip, 'crop_to_patches', False)
                and type(ip).__name__ != 'GotOcr2ImageProcessor'):
            if not hasattr(self, '_got_image_processor'):
                from transformers import AutoImageProcessor
                self._got_image_processor = (
                    AutoImageProcessor.from_pretrained(
                        self.ctx.model_config.model))
            if type(self._got_image_processor).__name__ == (
                    'GotOcr2ImageProcessor'):
                processor.image_processor = self._got_image_processor
        return processor

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}

    def _resize_output_size(
        self,
        *,
        height: int,
        width: int,
        max_len: int | None = None,
        min_len: int = 1,
        max_size: int | None = None,
    ) -> tuple[int, int]:
        # Set default value for max_len if not provided
        max_len = max(height, width) if max_len is None else max_len
        aspect_ratio = width / height

        # Handle the maximum size constraint
        if max_size is not None:
            max_len = min(max_len, max_size)

        # Adjust dimensions according to the aspect ratio
        if width >= height:
            width = max_len
            height = int(width / aspect_ratio)
        else:
            height = max_len
            width = int(height * aspect_ratio)

        # Ensure both width and height are even (if needed)
        height += height % 2
        width += width % 2

        # Ensure dimensions are not smaller than the minimum length
        height = max(height, min_len)
        width = max(width, min_len)

        return height, width

    def _get_resize_output_image_size(
        self,
        *,
        image_width: int,
        image_height: int,
        resolution_max_side: int,
    ) -> tuple[int, int]:
        hf_processor = self.get_hf_processor()
        image_processor: Idefics3ImageProcessor = hf_processor.image_processor
        # Handle both standard Idefics3 format and granite-docling format
        if "longest_edge" in image_processor.size:
            max_image_size = image_processor.size["longest_edge"]
        else:
            # Granite-docling format with height/width
            max_image_size = max(image_processor.size.get("height", 512),
                                image_processor.size.get("width", 512))

        if resolution_max_side > max_image_size:
            raise ValueError(
                "`resolution_max_side` cannot be larger than `max_image_size`"
            )

        height, width = image_height, image_width

        # Find the output size, when rescaling the longest edge to max_len and
        # preserving the aspect ratio
        height, width = self._resize_output_size(
            height=height, width=width, max_len=resolution_max_side
        )
        return height, width

    def _get_image_feature_grid_size(
        self,
        *,
        image_width: int,
        image_height: int,
        processor: Idefics3Processor,
        mm_kwargs: Mapping[str, object],
    ) -> tuple[int, int, int]:
        image_processor: Idefics3ImageProcessor = processor.image_processor

        # GotOcr2-style crop_to_patches: find optimal grid based on
        # aspect ratio, matching the GotOcr2ImageProcessor algorithm.
        if getattr(image_processor, 'crop_to_patches', False):
            grid_w, grid_h = self._get_crop_to_patches_grid_size(
                image_width=image_width,
                image_height=image_height,
                processor=processor,
            )
            return grid_w * grid_h + 1, grid_h, grid_w

        if "longest_edge" not in image_processor.size:
            # Granite-docling format with height/width - no tiling supported
            return 1, 0, 0

        return image_processor.get_number_of_image_patches(
            image_height,
            image_width,
            self.ctx.get_merged_mm_kwargs(mm_kwargs),
        )

    def _get_crop_to_patches_grid_size(
        self,
        *,
        image_width: int,
        image_height: int,
        processor: Idefics3Processor | None,
    ) -> tuple[int, int]:
        """GotOcr2-compatible optimal grid calculation for crop_to_patches."""
        if processor is None:
            processor = self.get_hf_processor()

        ip = processor.image_processor

        # Use the same get_optimal_tiled_canvas function that GotOcr2ImageProcessor
        # uses internally so that our patch count exactly matches what the processor
        # will produce (including the area-based tie-breaking logic).
        try:
            from transformers.models.got_ocr2.image_processing_got_ocr2 import (
                get_optimal_tiled_canvas,
            )
            max_patches = getattr(ip, 'max_patches', 16)
            min_patches = getattr(ip, 'min_patches', 1)
            patch_size = ip.size  # {"height": h, "width": w}
            num_columns, num_rows = get_optimal_tiled_canvas(
                (image_height, image_width),
                (patch_size["height"], patch_size["width"]),
                min_patches,
                max_patches,
            )
            if num_columns * num_rows <= 1:
                return (0, 0)
            return (num_columns, num_rows)
        except ImportError:
            pass

        # Fallback: naive aspect-ratio search (may not match GotOcr2 exactly)
        max_patches = getattr(ip, 'max_patches', 16)
        min_patches = getattr(ip, 'min_patches', 1)

        image_ar = image_width / image_height
        best_cols, best_rows = 1, 1
        best_diff = float('inf')

        for cols in range(1, max_patches + 1):
            for rows in range(1, max_patches + 1):
                total = cols * rows
                if total < min_patches or total > max_patches:
                    continue
                diff = abs(image_ar - cols / rows)
                if (diff < best_diff
                        or (diff == best_diff
                            and total > best_cols * best_rows)):
                    best_diff = diff
                    best_cols, best_rows = cols, rows

        if best_cols * best_rows <= 1:
            return (0, 0)
        return (best_cols, best_rows)

    def get_num_patches(
        self,
        *,
        image_width: int,
        image_height: int,
        processor: Idefics3Processor,
        mm_kwargs: Mapping[str, object],
    ) -> int:
        num_patches, _, _ = self._get_image_feature_grid_size(
            image_width=image_width,
            image_height=image_height,
            processor=processor,
            mm_kwargs=mm_kwargs,
        )

        return num_patches

    def _get_image_token(self, processor: Idefics3Processor) -> tuple[str, str, str]:
        image_token = processor.image_token
        fake_image_token = processor.fake_image_token
        global_image_token = processor.global_image_tag
        return image_token, fake_image_token, global_image_token

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
        grid_placeholder = "<row_{n_h}_col_{n_w}>"

        p_img = image_token * image_seq_len
        global_img_placeholder = fake_image_token + global_img_token + p_img
        tile_img_placeholder = fake_image_token + grid_placeholder + p_img

        _, grid_h, grid_w = self._get_image_feature_grid_size(
            image_width=image_width,
            image_height=image_height,
            processor=processor,
            mm_kwargs=mm_kwargs,
        )
        if grid_w == 0 and grid_h == 0:
            return global_img_placeholder + fake_image_token

        tiles_placeholder = list[str]()
        for i in range(grid_h):
            for j in range(grid_w):
                placeholder_per_tile = tile_img_placeholder.format(n_h=i + 1, n_w=j + 1)
                tiles_placeholder.append(placeholder_per_tile)
                # Add line break if it is the last tile in the row
                if j == grid_w - 1:
                    tiles_placeholder.append("\n")

        return "".join(
            [
                *tiles_placeholder,
                "\n",
                global_img_placeholder,
                fake_image_token,
            ]
        )

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


class Idefics3DummyInputsBuilder(BaseDummyInputsBuilder[Idefics3ProcessingInfo]):
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
        mm_processor_kwargs: Mapping[str, object] | None = None,
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        hf_processor = self.info.get_hf_processor(**(mm_processor_kwargs or {}))
        image_processor: Idefics3ImageProcessor = hf_processor.image_processor
        if (hasattr(image_processor, 'max_image_size')
                and isinstance(image_processor.max_image_size, dict)
                and "longest_edge" in image_processor.max_image_size):
            longest_edge = image_processor.max_image_size["longest_edge"]
        else:
            longest_edge = max(
                image_processor.size.get("height", 512),
                image_processor.size.get("width", 512),
            )

        image_overrides = mm_options.get("image") if mm_options else None

        return {
            "image": self._get_dummy_images(
                width=longest_edge,
                height=longest_edge,
                num_images=num_images,
                overrides=image_overrides,
            )
        }


class Idefics3MultiModalProcessor(BaseMultiModalProcessor[Idefics3ProcessingInfo]):
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        # Text-only input not supported in composite processor
        if not (images := mm_data.get("images", [])):
            prompt_ids = self.info.get_tokenizer().encode(prompt)
            prompt_ids = self._apply_hf_processor_tokens_only(prompt_ids)
            return BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")

        mm_kwargs = {"input_data_format": "channels_last", **mm_kwargs}
        processed_outputs = super()._call_hf_processor(
            prompt,
            mm_data,
            mm_kwargs,
            tok_kwargs,
        )

        hf_processor = self.info.get_hf_processor(**mm_kwargs)
        is_crop_to_patches = getattr(
            hf_processor.image_processor, 'crop_to_patches', False)

        if is_crop_to_patches:
            # GotOcr2ImageProcessor returns pixel_values as [total_patches, C, H, W]
            # with NO leading batch dimension. It also returns num_patches (total
            # patches per input image, including the thumbnail) in the BatchFeature.
            pv = processed_outputs["pixel_values"]

            # Use the num_patches the processor computed — it is accurate and avoids
            # discrepancies with get_optimal_tiled_canvas tie-breaking.
            num_patches_raw = processed_outputs.get("num_patches")
            if num_patches_raw is not None:
                if not isinstance(num_patches_raw, torch.Tensor):
                    num_patches = torch.tensor(num_patches_raw, dtype=torch.long)
                else:
                    num_patches = num_patches_raw.long()
            else:
                # Fallback: compute from image sizes
                mm_items = self.info.parse_mm_data(
                    {"image": images}, validate=False)
                parsed_images = mm_items.get_items("image", ImageProcessorItems)
                image_sizes = [
                    parsed_images.get_image_size(i)
                    for i in range(len(parsed_images))
                ]
                num_patches = torch.tensor([
                    self.info.get_num_patches(
                        image_width=s.width,
                        image_height=s.height,
                        processor=hf_processor,
                        mm_kwargs=mm_kwargs,
                    )
                    for s in image_sizes
                ])

            processed_outputs["num_patches"] = num_patches

            # GotOcr2 does not produce pixel_attention_mask; create an all-ones
            # mask of shape [total_patches, H, W].
            processed_outputs["pixel_attention_mask"] = torch.ones(
                pv.shape[0], pv.shape[-2], pv.shape[-1],
                dtype=torch.bool, device=pv.device,
            )
        else:
            # Standard Idefics3ImageProcessor returns pixel_values as
            # [batch=1, num_patches, C, H, W]; squeeze the batch dimension.
            mm_items = self.info.parse_mm_data(
                {"image": images}, validate=False)
            parsed_images = mm_items.get_items("image", ImageProcessorItems)
            image_sizes = [
                parsed_images.get_image_size(i)
                for i in range(len(parsed_images))
            ]
            num_patches = [
                self.info.get_num_patches(
                    image_width=size.width,
                    image_height=size.height,
                    processor=hf_processor,
                    mm_kwargs=mm_kwargs,
                )
                for size in image_sizes
            ]
            processed_outputs["num_patches"] = torch.tensor(num_patches)

            processed_outputs["pixel_values"].squeeze_(0)
            if "pixel_attention_mask" in processed_outputs:
                processed_outputs["pixel_attention_mask"].squeeze_(0)
            else:
                pv = processed_outputs["pixel_values"]
                processed_outputs["pixel_attention_mask"] = torch.ones(
                    pv.shape[0], pv.shape[-2], pv.shape[-1],
                    dtype=torch.bool, device=pv.device,
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
            pixel_attention_mask=MultiModalFieldConfig.flat_from_sizes(
                "image", num_patches
            ),
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

        def get_replacement_idefics3(item_idx: int) -> PromptUpdateDetails:
            images = mm_items.get_items("image", ImageProcessorItems)

            image_size = images.get_image_size(item_idx)

            image_repl = self.info.get_image_repl(
                image_width=image_size.width,
                image_height=image_size.height,
                processor=hf_processor,
                mm_kwargs=hf_processor_mm_kwargs,
            )

            return PromptUpdateDetails.select_text(
                image_repl,
                embed_text=image_token,
            )

        return [
            PromptReplacement(
                modality="image",
                target=image_token,
                replacement=get_replacement_idefics3,
            )
        ]


class Idefics3SimpleMLP(nn.Module):
    def __init__(
        self,
        config: Idefics3Config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        input_size = config.vision_config.hidden_size * (config.scale_factor**2)
        output_size = config.text_config.hidden_size
        self.proj = ReplicatedLinear(
            input_size,
            output_size,
            bias=False,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "proj"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.proj(x)
        return out


class Idefics3Connector(nn.Module):
    def __init__(
        self,
        config: Idefics3Config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.scale_factor = config.scale_factor
        self.modality_projection = Idefics3SimpleMLP(
            config,
            quant_config,
            prefix=maybe_prefix(prefix, "modality_projection"),
        )

    def pixel_shuffle(self, x: torch.Tensor, scale_factor: int = 2) -> torch.Tensor:
        bsz, seq, embed_dim = x.size()
        height = width = int(seq**0.5)
        x = x.view(bsz, height, width, embed_dim)
        x = x.view(bsz, height, int(width / scale_factor), embed_dim * scale_factor)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(
            bsz,
            int(width / scale_factor),
            int(height / scale_factor),
            embed_dim * (scale_factor**2),
        )
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(bsz, int(seq / (scale_factor**2)), embed_dim * (scale_factor**2))
        return x

    def forward(self, image_hidden_states: torch.Tensor) -> torch.Tensor:
        image_hidden_states = self.pixel_shuffle(image_hidden_states, self.scale_factor)
        image_hidden_states = self.modality_projection(image_hidden_states)
        return image_hidden_states


class Idefics3Model(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config: Idefics3Config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.vocab_size = self.config.text_config.vocab_size
        self.vision_model = Idefics3VisionTransformer(
            config.vision_config,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "vision_model"),
        )
        self.connector = Idefics3Connector(
            config,
            quant_config,
            prefix=maybe_prefix(prefix, "connector"),
        )

        # Check if text_config is GraniteMoeHybridConfig
        text_config = config.text_config
        is_granitemoehybrid = (
            getattr(text_config, "model_type", None) == "granitemoehybrid"
            or (hasattr(text_config, "layer_types")
                and text_config.layer_types is not None)
        )
        if is_granitemoehybrid:
            self._configure_granite_hybrid_cache(vllm_config, text_config)
            self.text_model = GraniteMoeHybridModel(
                vllm_config=vllm_config.with_hf_config(text_config),
                prefix=maybe_prefix(prefix, "text_model"),
            )
        else:
            self.text_model = LlamaModel(
                vllm_config=vllm_config.with_hf_config(text_config),
                prefix=maybe_prefix(prefix, "text_model"),
            )

        self.image_seq_len = int(
            ((config.vision_config.image_size // config.vision_config.patch_size) ** 2)
            / (config.scale_factor**2)
        )
        self.image_token_id = self.config.image_token_id

    @staticmethod
    def _configure_granite_hybrid_cache(vllm_config: VllmConfig,
                                        text_config) -> None:
        from vllm.model_executor.models.config import (
            HybridAttentionMambaModelConfig,
        )
        HybridAttentionMambaModelConfig.verify_and_update_config(vllm_config)

    def image_pixels_to_features(
        self,
        pixel_values: torch.Tensor,
        pixel_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        vision_dtype = (
            self.vision_model.embeddings.patch_embedding.weight.dtype
        )
        nb_values_per_image = pixel_values.shape[1:].numel()

        if pixel_values.dtype == torch.uint8:
            # uint8 path (recipe ⑧): images transferred as 1 byte/pixel;
            # rescale and normalize here on GPU.
            real_images_inds = pixel_values.sum(dim=(-1, -2, -3)) != 0
            if not real_images_inds.all():
                real_images_inds = (
                    (pixel_values == 0).sum(dim=(-1, -2, -3))
                    != nb_values_per_image
                )
            pixel_values = pixel_values[real_images_inds].contiguous()
            pixel_values = pixel_values.to(dtype=vision_dtype) / 255.0

            image_mean = getattr(
                self.config.vision_config, "image_mean", None)
            image_std = getattr(
                self.config.vision_config, "image_std", None)
            if image_mean is not None and image_std is not None:
                mean = torch.tensor(
                    image_mean, dtype=vision_dtype,
                    device=pixel_values.device,
                ).view(1, 3, 1, 1)
                std = torch.tensor(
                    image_std, dtype=vision_dtype,
                    device=pixel_values.device,
                ).view(1, 3, 1, 1)
                pixel_values = (pixel_values - mean) / std

            # GotOCR2 crops tiles to exact ViT resolution — every pixel is
            # valid, so skip unfold and pass None for patch_attention_mask.
            image_hidden_states = self.vision_model(
                pixel_values=pixel_values,
                patch_attention_mask=None,
            )
            return image_hidden_states

        # Float path (standard Idefics3)
        pixel_values = pixel_values.to(dtype=vision_dtype)

        # Remove padding images - padding images are full 0.
        real_images_inds = (pixel_values == 0.0).sum(
            dim=(-1, -2, -3)
        ) != nb_values_per_image
        pixel_values = pixel_values[real_images_inds].contiguous()

        # Handle the vision attention mask
        pixel_attention_mask = pixel_attention_mask[real_images_inds].contiguous()

        patch_size = self.config.vision_config.patch_size
        patches_subgrid = pixel_attention_mask.unfold(
            dimension=1, size=patch_size, step=patch_size
        )
        patches_subgrid = patches_subgrid.unfold(
            dimension=2, size=patch_size, step=patch_size
        )
        patch_attention_mask = (patches_subgrid.sum(dim=(-1, -2)) > 0).bool()

        image_hidden_states = self.vision_model(
            pixel_values=pixel_values,
            patch_attention_mask=patch_attention_mask,
        )

        return image_hidden_states

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.text_model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.text_model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states


@MULTIMODAL_REGISTRY.register_processor(
    Idefics3MultiModalProcessor,
    info=Idefics3ProcessingInfo,
    dummy_inputs=Idefics3DummyInputsBuilder,
)
class Idefics3ForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsLoRA,
    SupportsPP,
    HasInnerState,
    IsHybrid,
    SupportsMambaPrefixCaching,
):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<image>"

        raise ValueError("Only image modality is supported")

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig):
        text_config = vllm_config.model_config.hf_config.text_config
        temp_vllm_config = copy.deepcopy(vllm_config)
        temp_vllm_config.model_config.hf_config = text_config
        return GraniteMoeHybridForCausalLM.get_mamba_state_shape_from_config(
            temp_vllm_config)

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig):
        text_config = vllm_config.model_config.hf_config.text_config
        temp_vllm_config = copy.deepcopy(vllm_config)
        temp_vllm_config.model_config.hf_config = text_config
        return GraniteMoeHybridForCausalLM.get_mamba_state_dtype_from_config(
            temp_vllm_config)

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc,
                                                MambaStateCopyFunc]:
        return GraniteMoeHybridForCausalLM.get_mamba_state_copy_func()

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config

        with self._mark_composite_model(
            vllm_config,
            language_targets=(LlamaModel, GraniteMoeHybridModel),
            tower_targets={"image": (Idefics3VisionTransformer, Idefics3Connector)},
        ):
            self.model = Idefics3Model(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "model"),
            )

        self.image_token_id = self.config.image_token_id

        self.lm_head = ParallelLMHead(
            config.text_config.vocab_size,
            config.text_config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if self.config.text_config.tie_word_embeddings:
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

    def _parse_and_validate_image_input(self, **kwargs: object) -> ImageInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)

        if pixel_values is None and image_embeds is None:
            return None

        if image_embeds is not None:
            return Idefics3ImageEmbeddingInputs(
                type="image_embeds",
                data=image_embeds,
            )

        if pixel_values is not None:
            pixel_attention_mask = kwargs.pop("pixel_attention_mask")
            num_patches = kwargs.pop("num_patches")

            # Use actual pixel_values shape instead of config, to support models
            # like granite-docling that use different image sizes
            if pixel_values.ndim >= 3:
                expected_h = pixel_values.shape[-2]
                expected_w = pixel_values.shape[-1]
            else:
                expected_h = expected_w = self.config.vision_config.image_size

            return Idefics3ImagePixelInputs(
                type="pixel_values",
                pixel_values=pixel_values,
                pixel_attention_mask=pixel_attention_mask,
                num_patches=num_patches,
                resolve_bindings={"h": expected_h, "w": expected_w},
            )

        raise AssertionError("This line should be unreachable.")

    def _process_image_pixels(self, inputs: Idefics3ImagePixelInputs) -> torch.Tensor:
        pixel_values = inputs["pixel_values"]
        pixel_attention_mask = inputs["pixel_attention_mask"]

        return self.model.image_pixels_to_features(
            pixel_values,
            pixel_attention_mask=pixel_attention_mask,
        )

    def _process_image_input(
        self,
        image_input: ImageInputs,
    ) -> torch.Tensor | list[torch.Tensor]:
        if image_input["type"] == "image_embeds":
            return image_input["data"]

        image_features = self._process_image_pixels(image_input)
        image_features = self.model.connector(image_features)

        num_patches = image_input["num_patches"]
        return [e.flatten(0, 1) for e in image_features.split(num_patches.tolist())]

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return []

        return self._process_image_input(image_input)

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

        hidden_states = self.model.text_model(
            input_ids, positions, intermediate_tensors, inputs_embeds=inputs_embeds
        )

        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)

    def get_mm_mapping(self) -> MultiModelKeys:
        """
        Get the module prefix in multimodal models
        """
        return MultiModelKeys.from_string_field(
            language_model="model.text_model",
            connector="model.connector",
            tower_model="model.vision_model",
        )

    def get_num_mm_encoder_tokens(
        self,
        num_image_tokens: int,
    ) -> int:
        hf_config = self.config
        scale_factor = hf_config.scale_factor

        return num_image_tokens * scale_factor**2

    def get_num_mm_connector_tokens(
        self,
        num_vision_tokens: int,
    ) -> int:
        hf_config = self.config
        scale_factor = hf_config.scale_factor

        return num_vision_tokens // scale_factor**2
