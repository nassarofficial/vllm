# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for GraniteForDocling's multimodal preprocessing."""

import pytest
import torch

from vllm.multimodal import MULTIMODAL_REGISTRY

from ....conftest import ImageTestAssets
from ...registry import HF_EXAMPLE_MODELS
from ...utils import build_model_context


@pytest.mark.parametrize("model_id", ["docling-project/granite-for-docling-500m"])
@pytest.mark.parametrize("fine_route", [False, True])
@pytest.mark.parametrize("num_imgs", [1, 2])
@pytest.mark.parametrize("kwargs_on_init", [True, False])
def test_processor_fine_route(
    image_assets: ImageTestAssets,
    model_id: str,
    fine_route: bool,
    num_imgs: int,
    kwargs_on_init: bool,
):
    """The prompt expansion, the tile mask and the number of image tokens agree
    for both connector paths, whether the route is set at init or per request.
    """
    HF_EXAMPLE_MODELS.find_hf_info(model_id).check_available_online(on_fail="skip")

    mm_processor_kwargs = {"fine_route": fine_route}
    ctx = build_model_context(
        model_id,
        mm_processor_kwargs=mm_processor_kwargs if kwargs_on_init else None,
        limit_mm_per_prompt={"image": num_imgs},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    hf_processor_mm_kwargs = {} if kwargs_on_init else mm_processor_kwargs
    hf_processor = processor.info.get_hf_processor()
    image_processor = hf_processor.image_processor

    # A 2:1 page is tiled on a 2 x 4 grid plus the thumbnail
    tile_size = image_processor.size["height"]
    image = image_assets[0].pil_image.resize((2 * tile_size, 4 * tile_size))
    mm_data = {"image": [image] * num_imgs}
    prompt = hf_processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}] * num_imgs}],
        add_generation_prompt=True,
    )

    processed_inputs = processor(
        prompt,
        mm_items=processor.info.parse_mm_data(mm_data),
        hf_processor_mm_kwargs=hf_processor_mm_kwargs,
    )
    mm_kwargs = processed_inputs["mm_kwargs"].get_data()

    num_tiles = 2 * 4 + 1
    image_seq_len = hf_processor.image_seq_len * (4 if fine_route else 1)
    image_token_id = hf_processor.image_token_id
    num_image_tokens = processed_inputs["prompt_token_ids"].count(image_token_id)
    assert num_image_tokens == num_imgs * num_tiles * image_seq_len
    assert mm_kwargs["pixel_values"].shape[:2] == (num_imgs * num_tiles, 3)
    assert mm_kwargs["num_patches"].tolist() == [num_tiles] * num_imgs
    assert torch.equal(
        mm_kwargs["tile_fine_mask"],
        torch.full((num_imgs * num_tiles,), fine_route, dtype=torch.bool),
    )
