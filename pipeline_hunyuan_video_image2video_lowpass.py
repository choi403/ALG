# Copyright 2024 The HunyuanVideo Team and The HuggingFace Team. All rights reserved.
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

import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    LlamaTokenizerFast,
    LlavaForConditionalGeneration,
)

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.loaders import HunyuanVideoLoraLoaderMixin
from diffusers.models import AutoencoderKLHunyuanVideo, HunyuanVideoTransformer3DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import is_torch_xla_available, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.hunyuan_video.pipeline_output import HunyuanVideoPipelineOutput
import math
import torchvision.transforms.functional as tvF
import torch.nn.functional as F

import lp_utils

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        >>> import torch
        >>> from diffusers import HunyuanVideoImageToVideoPipeline, HunyuanVideoTransformer3DModel
        >>> from diffusers.utils import load_image, export_to_video

        >>> # Available checkpoints: hunyuanvideo-community/HunyuanVideo-I2V, hunyuanvideo-community/HunyuanVideo-I2V-33ch
        >>> model_id = "hunyuanvideo-community/HunyuanVideo-I2V"
        >>> transformer = HunyuanVideoTransformer3DModel.from_pretrained(
        ...     model_id, subfolder="transformer", torch_dtype=torch.bfloat16
        ... )
        >>> pipe = HunyuanVideoImageToVideoPipeline.from_pretrained(
        ...     model_id, transformer=transformer, torch_dtype=torch.float16
        ... )
        >>> pipe.vae.enable_tiling()
        >>> pipe.to("cuda")

        >>> prompt = "A man with short gray hair plays a red electric guitar."
        >>> image = load_image(
        ...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/guitar-man.png"
        ... )

        >>> # If using hunyuanvideo-community/HunyuanVideo-I2V
        >>> output = pipe(image=image, prompt=prompt, guidance_scale=6.0).frames[0]

        >>> # If using hunyuanvideo-community/HunyuanVideo-I2V-33ch
        >>> output = pipe(image=image, prompt=prompt, guidance_scale=1.0, true_cfg_scale=1.0).frames[0]

        >>> export_to_video(output, "output.mp4", fps=15)
        ```
"""


DEFAULT_PROMPT_TEMPLATE = {
    "template": (
        "<|start_header_id|>system<|end_header_id|>\n\n<image>\nDescribe the video by detailing the following aspects according to the reference image: "
        "1. The main content and theme of the video."
        "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects."
        "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects."
        "4. background environment, light, style and atmosphere."
        "5. camera angles, movements, and transitions used in the video:<|eot_id|>\n\n"
        "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    ),
    "crop_start": 103,
    "image_emb_start": 5,
    "image_emb_end": 581,
    "image_emb_len": 576,
    "double_return_token_id": 271,
}


def _expand_input_ids_with_image_tokens(
    text_input_ids,
    prompt_attention_mask,
    max_sequence_length,
    image_token_index,
    image_emb_len,
    image_emb_start,
    image_emb_end,
    pad_token_id,
):
    special_image_token_mask = text_input_ids == image_token_index
    num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)
    batch_indices, non_image_indices = torch.where(text_input_ids != image_token_index)

    max_expanded_length = max_sequence_length + (num_special_image_tokens.max() * (image_emb_len - 1))
    new_token_positions = torch.cumsum((special_image_token_mask * (image_emb_len - 1) + 1), -1) - 1
    text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

    expanded_input_ids = torch.full(
        (text_input_ids.shape[0], max_expanded_length),
        pad_token_id,
        dtype=text_input_ids.dtype,
        device=text_input_ids.device,
    )
    expanded_input_ids[batch_indices, text_to_overwrite] = text_input_ids[batch_indices, non_image_indices]
    expanded_input_ids[batch_indices, image_emb_start:image_emb_end] = image_token_index

    expanded_attention_mask = torch.zeros(
        (text_input_ids.shape[0], max_expanded_length),
        dtype=prompt_attention_mask.dtype,
        device=prompt_attention_mask.device,
    )
    attn_batch_indices, attention_indices = torch.where(expanded_input_ids != pad_token_id)
    expanded_attention_mask[attn_batch_indices, attention_indices] = 1.0
    expanded_attention_mask = expanded_attention_mask.to(prompt_attention_mask.dtype)
    position_ids = (expanded_attention_mask.cumsum(-1) - 1).masked_fill_((expanded_attention_mask == 0), 1)

    return {
        "input_ids": expanded_input_ids,
        "attention_mask": expanded_attention_mask,
        "position_ids": position_ids,
    }



def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


class HunyuanVideoImageToVideoPipeline(DiffusionPipeline, HunyuanVideoLoraLoaderMixin):
    r"""
    Pipeline for image-to-video generation using HunyuanVideo.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        text_encoder ([`LlavaForConditionalGeneration`]):
            [Llava Llama3-8B](https://huggingface.co/xtuner/llava-llama-3-8b-v1_1-transformers).
        tokenizer (`LlamaTokenizer`):
            Tokenizer from [Llava Llama3-8B](https://huggingface.co/xtuner/llava-llama-3-8b-v1_1-transformers).
        transformer ([`HunyuanVideoTransformer3DModel`]):
            Conditional Transformer to denoise the encoded image latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKLHunyuanVideo`]):
            Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
        text_encoder_2 ([`CLIPTextModel`]):
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer_2 (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/en/model_doc/clip#transformers.CLIPTokenizer).
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        text_encoder: LlavaForConditionalGeneration,
        tokenizer: LlamaTokenizerFast,
        transformer: HunyuanVideoTransformer3DModel,
        vae: AutoencoderKLHunyuanVideo,
        scheduler: FlowMatchEulerDiscreteScheduler,
        text_encoder_2: CLIPTextModel,
        tokenizer_2: CLIPTokenizer,
        image_processor: CLIPImageProcessor,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            image_processor=image_processor,
        )

        self.vae_scaling_factor = self.vae.config.scaling_factor if getattr(self, "vae", None) else 0.476986
        self.vae_scale_factor_temporal = self.vae.temporal_compression_ratio if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = self.vae.spatial_compression_ratio if getattr(self, "vae", None) else 8
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

    def _get_llama_prompt_embeds(
        self,
        image: torch.Tensor,
        prompt: Union[str, List[str]],
        prompt_template: Dict[str, Any],
        num_videos_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        max_sequence_length: int = 256,
        num_hidden_layers_to_skip: int = 2,
        image_embed_interleave: int = 2,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_template["template"].format(p) for p in prompt]

        crop_start = prompt_template.get("crop_start", None)

        image_emb_len = prompt_template.get("image_emb_len", 576)
        image_emb_start = prompt_template.get("image_emb_start", 5)
        image_emb_end = prompt_template.get("image_emb_end", 581)
        double_return_token_id = prompt_template.get("double_return_token_id", 271)

        if crop_start is None:
            prompt_template_input = self.tokenizer(
                prompt_template["template"],
                padding="max_length",
                return_tensors="pt",
                return_length=False,
                return_overflowing_tokens=False,
                return_attention_mask=False,
            )
            crop_start = prompt_template_input["input_ids"].shape[-1]
            # Remove <|start_header_id|>, <|end_header_id|>, assistant, <|eot_id|>, and placeholder {}
            crop_start -= 5

        max_sequence_length += crop_start
        text_inputs = self.tokenizer(
            prompt,
            max_length=max_sequence_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_length=False,
            return_overflowing_tokens=False,
            return_attention_mask=True,
        )
        text_input_ids = text_inputs.input_ids.to(device=device)
        prompt_attention_mask = text_inputs.attention_mask.to(device=device)

        image_embeds = self.image_processor(image, return_tensors="pt").pixel_values.to(device)

        image_token_index = self.text_encoder.config.image_token_index
        pad_token_id = self.text_encoder.config.pad_token_id
        expanded_inputs = _expand_input_ids_with_image_tokens(
            text_input_ids,
            prompt_attention_mask,
            max_sequence_length,
            image_token_index,
            image_emb_len,
            image_emb_start,
            image_emb_end,
            pad_token_id,
        )
        prompt_embeds = self.text_encoder(
            **expanded_inputs,
            pixel_values=image_embeds,
            output_hidden_states=True,
        ).hidden_states[-(num_hidden_layers_to_skip + 1)]
        prompt_embeds = prompt_embeds.to(dtype=dtype)

        if crop_start is not None and crop_start > 0:
            text_crop_start = crop_start - 1 + image_emb_len
            batch_indices, last_double_return_token_indices = torch.where(text_input_ids == double_return_token_id)

            if last_double_return_token_indices.shape[0] == 3:
                # in case the prompt is too long
                last_double_return_token_indices = torch.cat(
                    (last_double_return_token_indices, torch.tensor([text_input_ids.shape[-1]]))
                )
                batch_indices = torch.cat((batch_indices, torch.tensor([0])))

            last_double_return_token_indices = last_double_return_token_indices.reshape(text_input_ids.shape[0], -1)[
                :, -1
            ]
            batch_indices = batch_indices.reshape(text_input_ids.shape[0], -1)[:, -1]
            assistant_crop_start = last_double_return_token_indices - 1 + image_emb_len - 4
            assistant_crop_end = last_double_return_token_indices - 1 + image_emb_len
            attention_mask_assistant_crop_start = last_double_return_token_indices - 4
            attention_mask_assistant_crop_end = last_double_return_token_indices

            prompt_embed_list = []
            prompt_attention_mask_list = []
            image_embed_list = []
            image_attention_mask_list = []

            for i in range(text_input_ids.shape[0]):
                prompt_embed_list.append(
                    torch.cat(
                        [
                            prompt_embeds[i, text_crop_start : assistant_crop_start[i].item()],
                            prompt_embeds[i, assistant_crop_end[i].item() :],
                        ]
                    )
                )
                prompt_attention_mask_list.append(
                    torch.cat(
                        [
                            prompt_attention_mask[i, crop_start : attention_mask_assistant_crop_start[i].item()],
                            prompt_attention_mask[i, attention_mask_assistant_crop_end[i].item() :],
                        ]
                    )
                )
                image_embed_list.append(prompt_embeds[i, image_emb_start:image_emb_end])
                image_attention_mask_list.append(
                    torch.ones(image_embed_list[-1].shape[0]).to(prompt_embeds.device).to(prompt_attention_mask.dtype)
                )

            prompt_embed_list = torch.stack(prompt_embed_list)
            prompt_attention_mask_list = torch.stack(prompt_attention_mask_list)
            image_embed_list = torch.stack(image_embed_list)
            image_attention_mask_list = torch.stack(image_attention_mask_list)

            if 0 < image_embed_interleave < 6:
                image_embed_list = image_embed_list[:, ::image_embed_interleave, :]
                image_attention_mask_list = image_attention_mask_list[:, ::image_embed_interleave]

            assert (
                prompt_embed_list.shape[0] == prompt_attention_mask_list.shape[0]
                and image_embed_list.shape[0] == image_attention_mask_list.shape[0]
            )

            prompt_embeds = torch.cat([image_embed_list, prompt_embed_list], dim=1)
            prompt_attention_mask = torch.cat([image_attention_mask_list, prompt_attention_mask_list], dim=1)

        return prompt_embeds, prompt_attention_mask

    def _get_clip_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_videos_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        max_sequence_length: int = 77,
    ) -> torch.Tensor:
        device = device or self._execution_device
        dtype = dtype or self.text_encoder_2.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt

        text_inputs = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer_2(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer_2.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder_2(text_input_ids.to(device), output_hidden_states=False).pooler_output
        return prompt_embeds

    def encode_prompt(
        self,
        image: torch.Tensor,
        prompt: Union[str, List[str]],
        prompt_2: Union[str, List[str]] = None,
        prompt_template: Dict[str, Any] = DEFAULT_PROMPT_TEMPLATE,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        max_sequence_length: int = 256,
        image_embed_interleave: int = 2,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if prompt_embeds is None:
            prompt_embeds, prompt_attention_mask = self._get_llama_prompt_embeds(
                image,
                prompt,
                prompt_template,
                num_videos_per_prompt,
                device=device,
                dtype=dtype,
                max_sequence_length=max_sequence_length,
                image_embed_interleave=image_embed_interleave,
            )

        if pooled_prompt_embeds is None:
            if prompt_2 is None:
                prompt_2 = prompt
            pooled_prompt_embeds = self._get_clip_prompt_embeds(
                prompt,
                num_videos_per_prompt,
                device=device,
                dtype=dtype,
                max_sequence_length=77,
            )

        return prompt_embeds, pooled_prompt_embeds, prompt_attention_mask

    def check_inputs(
        self,
        prompt,
        prompt_2,
        height,
        width,
        prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        prompt_template=None,
        true_cfg_scale=1.0,
        guidance_scale=1.0,
    ):
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 16 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt_2 is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt_2`: {prompt_2} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif prompt_2 is not None and (not isinstance(prompt_2, str) and not isinstance(prompt_2, list)):
            raise ValueError(f"`prompt_2` has to be of type `str` or `list` but is {type(prompt_2)}")

        if prompt_template is not None:
            if not isinstance(prompt_template, dict):
                raise ValueError(f"`prompt_template` has to be of type `dict` but is {type(prompt_template)}")
            if "template" not in prompt_template:
                raise ValueError(
                    f"`prompt_template` has to contain a key `template` but only found {prompt_template.keys()}"
                )

        if true_cfg_scale > 1.0 and guidance_scale > 1.0:
            logger.warning(
                "Both `true_cfg_scale` and `guidance_scale` are greater than 1.0. This will result in both "
                "classifier-free guidance and embedded-guidance to be applied. This is not recommended "
                "as it may lead to higher memory usage, slower inference and potentially worse results."
            )

    def prepare_latents(
        self,
        image: torch.Tensor,
        batch_size: int,
        num_channels_latents: int = 32,
        height: int = 720,
        width: int = 1280,
        num_frames: int = 129,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        image_condition_type: str = "latent_concat",
        i2v_stable: bool = False,
    ) -> torch.Tensor:
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height, latent_width = height // self.vae_scale_factor_spatial, width // self.vae_scale_factor_spatial
        shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, latent_width)

        image = image.unsqueeze(2)  # [B, C, 1, H, W]
        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(self.vae.encode(image[i].unsqueeze(0)), generator[i], "argmax")
                for i in range(batch_size)
            ]
        else:
            image_latents = [retrieve_latents(self.vae.encode(img.unsqueeze(0)), generator, "argmax") for img in image]

        image_latents = torch.cat(image_latents, dim=0).to(dtype) * self.vae_scaling_factor

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        if i2v_stable:
            image_latents = image_latents.repeat(1, 1, num_latent_frames, 1, 1)
            t = torch.tensor([0.999]).to(device=device)
            latents = latents * t + image_latents * (1 - t)

        if image_condition_type == "token_replace":
            image_latents = image_latents[:, :, :1]

        return latents, image_latents

    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        self.vae.enable_tiling()

    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_tiling()

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    def prepare_lp(
        self,
        # --- Filter Selection & Strength ---
        lp_filter_type: str,
        lp_blur_sigma: float,
        lp_blur_kernel_size: float,
        lp_resize_factor: float,
        # --- Contextual Info ---
        generator: torch.Generator,
        num_frames: int,
        use_low_pass_guidance: bool,
        lp_filter_in_latent: bool,
        # --- Inputs to filter ---
        orig_image_latents: torch.Tensor,
        orig_image_tensor: torch.Tensor,
        last_image: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """
        Prepares a low-pass filtered version of the initial image condition for guidance. (HunyuanVideo)

        This function works in two modes:
        1.  **Filtering in Image (RGB) Space (`lp_filter_in_latent=False`)**:
            It applies a low-pass filter to the source image, constructs a video tensor (e.g., first frame is
            the filtered image, last frame is an optionally provided filtered `last_image`, and the rest are zeros),
            encodes this video tensor with the VAE, normalizes the result, and finally prepends a temporal mask
            to create a condition tensor in the format expected by the transformer (`[mask, latents]`).
        2.  **Filtering in Latent Space (`lp_filter_in_latent=True`)**:
            Directly applies the low-pass filter to the already-encoded `orig_image_latents`.

        Args:
            lp_filter_type (`str`): The type of low-pass filter to apply, e.g., 'gaussian_blur', 'down_up'.
            lp_blur_sigma (`float`): The sigma value for the Gaussian blur filter.
            lp_blur_kernel_size (`float`): The kernel size for the Gaussian blur filter.
            lp_resize_factor (`float`): The resizing factor for the 'down_up' filter.
            generator (`torch.Generator`): A random generator, used for VAE sampling when filtering in image space.
            num_frames (`int`): The target number of frames for the video condition tensor.
            use_low_pass_guidance (`bool`): If `False`, the function returns `None` immediately.
            lp_filter_in_latent (`bool`): If `True`, filtering is applied in latent space. Otherwise, in image space.
            orig_image_latents (`torch.Tensor`): The VAE-encoded latents of the original image. Used when
                `lp_filter_in_latent` is `True`.
            orig_image_tensor (`torch.Tensor`): The preprocessed original image tensor (RGB). Used when
                `lp_filter_in_latent` is `False`.
            last_image (`Optional[torch.Tensor]`, defaults to `None`):
                An optional image tensor for the last frame. If provided (and when filtering in image space), it will
                also be low-pass filtered and used as the last frame of the VAE input.

        Returns:
            `Optional[torch.Tensor]`: A tensor containing the low-pass filtered image condition ready for the
            transformer, or `None` if `use_low_pass_guidance` is `False`.
        """
        if not use_low_pass_guidance:
            return None

        if not lp_filter_in_latent:
            # --- Filter in Image (RGB) Space ---
            # 1. Apply the low-pass filter to the source image(s).
            image_lp = lp_utils.apply_low_pass_filter(
                orig_image_tensor,
                filter_type=lp_filter_type,
                blur_sigma=lp_blur_sigma,
                blur_kernel_size=lp_blur_kernel_size,
                resize_factor=lp_resize_factor,
            )
            image_lp_vae_input = image_lp.unsqueeze(2)

            batch_size,_,height,width = orig_image_tensor.shape
            latent_height = height // self.vae_scale_factor_spatial
            latent_width = width // self.vae_scale_factor_spatial

            # 2. Construct a video tensor to be encoded. This tensor has the filtered image as the first frame.
            # If a `last_image` is given, it's also filtered and placed at the end. Intermediate frames are black.
            if last_image is None:
                video_condition = torch.cat(
                    [image_lp_vae_input, image_lp_vae_input.new_zeros(image_lp_vae_input.shape[0], image_lp_vae_input.shape[1], num_frames - 1, height, width)], dim=2
                )
            else:

                last_image_lp = lp_utils.apply_low_pass_filter(
                    last_image,
                    filter_type=lp_filter_type,
                    blur_sigma=lp_blur_sigma,
                    blur_kernel_size=lp_blur_kernel_size,
                    resize_factor=lp_resize_factor,
                )

                last_image_lp = last_image_lp.unsqueeze(2)
                video_condition = torch.cat(
                    [image_lp_vae_input, image_lp_vae_input.new_zeros(image_lp_vae_input.shape[0], image_lp_vae_input.shape[1], num_frames - 2, height, width), last_image_lp],
                    dim=2,
                )
            # 3. Encode the constructed video tensor and normalize the resulting latents.
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(image_lp.device, image_lp.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                image_lp.device, image_lp.dtype
            )
            encoded_lp = self.vae.encode(video_condition).latent_dist.sample(generator=generator)
            latent_condition = (encoded_lp - latents_mean) * latents_std

            # 4. Create a temporal mask. The transformer condition is `[mask, latents]`.
            # The mask is 1 for conditioned frames (first, and optionally last) and 0 for unconditioned frames.
            mask_lat_size = torch.ones(batch_size, 1, num_frames, latent_height, latent_width)

            if last_image is None:
                mask_lat_size[:, :, list(range(1, num_frames))] = 0
            else:
                mask_lat_size[:, :, list(range(1, num_frames - 1))] = 0
            first_frame_mask = mask_lat_size[:, :, 0:1]
            first_frame_mask = torch.repeat_interleave(first_frame_mask, dim=2, repeats=self.vae_scale_factor_temporal)
            mask_lat_size = torch.concat([first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2)
            mask_lat_size = mask_lat_size.view(batch_size, -1, self.vae_scale_factor_temporal, latent_height, latent_width)
            mask_lat_size = mask_lat_size.transpose(1, 2)
            mask_lat_size = mask_lat_size.to(latent_condition.device)

            # 5. Concatenate the mask and the normalized latents along the channel dimension.
            lp_image_latents = torch.concat([mask_lat_size, latent_condition], dim=1)

        else:
            # --- Filter Directly in Latent Space ---
            # This path assumes `orig_image_latents` is already prepared and just needs filtering.
            lp_image_latents = lp_utils.apply_low_pass_filter(
                orig_image_latents,
                filter_type=lp_filter_type,
                blur_sigma=lp_blur_sigma,
                blur_kernel_size=lp_blur_kernel_size,
                resize_factor=lp_resize_factor,
            )

            if self.transformer.config.patch_size is not None:
                remainder = lp_image_latents.size(1) % self.transformer.config.patch_size
                if remainder != 0:
                    num_to_prepend = self.transformer.config.patch_size - remainder
                    num_to_prepend = min(num_to_prepend, lp_image_latents.shape[1])
                    first_frames_to_prepend = lp_image_latents[:, :num_to_prepend, ...]
                    lp_image_latents = torch.cat([first_frames_to_prepend, lp_image_latents], dim=1)


        lp_image_latents = lp_image_latents.to(dtype=orig_image_latents.dtype)

        return lp_image_latents

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        image: PIL.Image.Image,
        prompt: Union[str, List[str]] = None,
        prompt_2: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = "bad quality",
        negative_prompt_2: Union[str, List[str]] = None,
        height: int = 720,
        width: int = 1280,
        num_frames: int = 129,
        num_inference_steps: int = 50,
        sigmas: List[float] = None,
        true_cfg_scale: float = 1.0,
        guidance_scale: float = 1.0,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        prompt_template: Dict[str, Any] = DEFAULT_PROMPT_TEMPLATE,
        max_sequence_length: int = 256,
        image_embed_interleave: Optional[int] = None,

        use_low_pass_guidance: bool = False,
        lp_filter_type: str = "none", # {'gaussian_blur', 'down_up'}
        lp_filter_in_latent: bool = False, # When set to True, low-pass filter is done after encoder. If False, low-pass filter is applied to image directly before encoder.
        lp_blur_sigma: float = 15.0, # Used with 'gaussian_blur'. Gaussian filter sigma value.
        lp_blur_kernel_size: float = 0.02734375, # Used with 'gaussian_blur'. Gaussian filter size. When set to int, used directly as kernel size. When set to float, H * `lp_blur_kernel_size` is used as kernel size.
        lp_resize_factor: float = 0.25, # Used with 'down_up'. Image is bilinearly downsized to (`lp_resize_factor` * WIDTH, `lp_resize_factor` * HEIGHT) and then back to original.

        lp_strength_schedule_type: str = "none",  # Scheduling type for low-pass filtering strength. Options: {"none", "linear", "interval", "exponential"}
        schedule_blur_kernel_size: bool = False,  # If True, schedule blur kernel size as well. Otherwise, fix to initial value.

        # --- Constant Interval Scheduling Params for LP Strength ---
        schedule_interval_start_time: float = 0.0,    # Starting timestep for interval scheduling
        schedule_interval_end_time: float = 0.05,     # Ending timestep for interval scheduling

        # --- Linear Scheduling Params for LP Strength ---
        schedule_linear_start_weight: float = 1.0,    # Starting LP weight for linear scheduling at t=T (step 0)
        schedule_linear_end_weight: float = 0.0,      # Ending LP weight for linear scheduling at t=T * schedule_linear_end_time
        schedule_linear_end_time: float = 0.5,  # Timestep fraction at which schedule_linear_end is reached

        # --- Exponential Scheduling Params for LP Strength ---
        schedule_exp_decay_rate: float = 10.0, # Decay rate for 'exponential' schedule. Higher values decay faster. Strength = exp(-rate * time_fraction).

        lp_on_noisy_latent = False,
        enable_lp_img_embeds = False,
        i2v_stable= False,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                will be used instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `true_cfg_scale` is
                not greater than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in all the text-encoders.
            height (`int`, defaults to `720`):
                The height in pixels of the generated image.
            width (`int`, defaults to `1280`):
                The width in pixels of the generated image.
            num_frames (`int`, defaults to `129`):
                The number of frames in the generated video.
            num_inference_steps (`int`, defaults to `50`):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            true_cfg_scale (`float`, *optional*, defaults to 1.0):
                When > 1.0 and a provided `negative_prompt`, enables true classifier-free guidance.
            guidance_scale (`float`, defaults to `1.0`):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality. Note that the only available
                HunyuanVideo model is CFG-distilled, which means that traditional guidance between unconditional and
                conditional latent is not applied.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`HunyuanVideoPipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            prompt_template (`Dict[str, Any]`, *optional*, defaults to `DEFAULT_PROMPT_TEMPLATE`):
                A dictionary defining the template for constructing the LLaVA prompt. It should include keys like
                `"template"`, `"crop_start"`, `"image_emb_start"`, `"image_emb_end"`, `"image_emb_len"`, and
                `"double_return_token_id"`.
            max_sequence_length (`int`, *optional*, defaults to 256):
                The maximum sequence length for the LLaVA text encoder.
            image_embed_interleave (`int`, *optional*):
                The interleave factor for image embeddings. Defaults to 2 if `image_condition_type` is
                `"latent_concat"`, 4 if `"token_replace"`, otherwise 1.
            use_low_pass_guidance (`bool`, *optional*, defaults to `False`):
                Whether to use low-pass guidance. This can help to improve the temporal consistency of the generated
                video.
            lp_filter_type (`str`, *optional*, defaults to `"none"`):
                The type of low-pass filter to apply. Can be one of `gaussian_blur` or `down_up`.
            lp_filter_in_latent (`bool`, *optional*, defaults to `False`):
                If `True`, the low-pass filter is applied to the latent representation of the image. If `False`, it is
                applied to the image in pixel space before encoding.
            lp_blur_sigma (`float`, *optional*, defaults to `15.0`):
                The sigma value for the Gaussian blur filter. Only used if `lp_filter_type` is `gaussian_blur`.
            lp_blur_kernel_size (`float`, *optional*, defaults to `0.02734375`):
                The kernel size for the Gaussian blur filter. If an `int`, it's used directly. If a `float`, the kernel
                size is calculated as `height * lp_blur_kernel_size`. Only used if `lp_filter_type` is `gaussian_blur`.
            lp_resize_factor (`float`, *optional*, defaults to `0.25`):
                The resize factor for the down-sampling and up-sampling filter. Only used if `lp_filter_type` is
                `down_up`.
            lp_strength_schedule_type (`str`, *optional*, defaults to `"none"`):
                The scheduling type for the low-pass filter strength. Can be one of `none`, `linear`, `interval`, or
                `exponential`.
            schedule_blur_kernel_size (`bool`, *optional*, defaults to `False`):
                If `True`, the blur kernel size is also scheduled along with the strength. Otherwise, it remains fixed.
            schedule_interval_start_time (`float`, *optional*, defaults to `0.0`):
                The starting timestep fraction for interval scheduling. Only used if `lp_strength_schedule_type` is
                `interval`.
            schedule_interval_end_time (`float`, *optional*, defaults to `0.05`):
                The ending timestep fraction for interval scheduling. Only used if `lp_strength_schedule_type` is
                `interval`.
            schedule_linear_start_weight (`float`, *optional*, defaults to `1.0`):
                The starting weight for the low-pass filter strength in a linear schedule. Corresponds to the first
                timestep. Only used if `lp_strength_schedule_type` is `linear`.
            schedule_linear_end_weight (`float`, *optional*, defaults to `0.0`):
                The ending weight for the low-pass filter strength in a linear schedule. Only used if
                `lp_strength_schedule_type` is `linear`.
            schedule_linear_end_time (`float`, *optional*, defaults to `0.5`):
                The timestep fraction at which `schedule_linear_end_weight` is reached in a linear schedule. Only used
                if `lp_strength_schedule_type` is `linear`.
            schedule_exp_decay_rate (`float`, *optional*, defaults to `10.0`):
                The decay rate for the exponential schedule. Higher values lead to faster decay. Only used if
                `lp_strength_schedule_type` is `exponential`.
            lp_on_noisy_latent (`bool`, *optional*, defaults to `False`):
                If `True` and using low-pass guidance with true CFG, applies the low-pass condition to the noisy latent input
                when the low-pass strength is zero, instead of using the original image condition.
            enable_lp_img_embeds (`bool`, *optional*, defaults to `False`):
                Whether to apply low-pass filtering to image embeddings.
            i2v_stable (`bool`, *optional*, defaults to `False`):
                If `True`, initializes the video latents with initial image latents.

        Examples:

        Returns:
            [`~HunyuanVideoPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`HunyuanVideoPipelineOutput`] is returned, otherwise a `tuple` is returned
                where the first element is a list with the generated images and the second element is a list of `bool`s
                indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.
        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            prompt_embeds,
            callback_on_step_end_tensor_inputs,
            prompt_template,
            true_cfg_scale,
            guidance_scale,
        )

        image_condition_type = self.transformer.config.image_condition_type
        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_pooled_prompt_embeds is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        image_embed_interleave = (
            image_embed_interleave
            if image_embed_interleave is not None
            else (
                2 if image_condition_type == "latent_concat" else 4 if image_condition_type == "token_replace" else 1
            )
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self._execution_device

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Prepare latent variables
        vae_dtype = self.vae.dtype
        image_tensor = self.video_processor.preprocess(image, height, width).to(device, vae_dtype)

        if image_condition_type == "latent_concat":
            num_channels_latents = (self.transformer.config.in_channels - 1) // 2
        elif image_condition_type == "token_replace":
            num_channels_latents = self.transformer.config.in_channels

        latents, image_latents = self.prepare_latents(
            image_tensor,
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            torch.float32,
            device,
            generator,
            latents,
            image_condition_type,
            i2v_stable
        )
        if image_condition_type == "latent_concat":
            image_latents[:, :, 1:] = 0
            mask = image_latents.new_ones(image_latents.shape[0], 1, *image_latents.shape[2:])
            mask[:, :, 1:] = 0

        # 4. Encode input prompt
        transformer_dtype = self.transformer.dtype
        prompt_embeds, pooled_prompt_embeds, prompt_attention_mask = self.encode_prompt(
            image=image,
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_template=prompt_template,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            device=device,
            max_sequence_length=max_sequence_length,
            image_embed_interleave=image_embed_interleave,
        )
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        prompt_attention_mask = prompt_attention_mask.to(transformer_dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(transformer_dtype)

        if do_true_cfg:
            black_image = PIL.Image.new("RGB", (width, height), 0)
            negative_prompt_embeds, negative_pooled_prompt_embeds, negative_prompt_attention_mask = self.encode_prompt(
                image=black_image,
                prompt=negative_prompt,
                prompt_2=negative_prompt_2,
                prompt_template=prompt_template,
                num_videos_per_prompt=num_videos_per_prompt,
                prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=negative_pooled_prompt_embeds,
                prompt_attention_mask=negative_prompt_attention_mask,
                device=device,
                max_sequence_length=max_sequence_length,
                image_embed_interleave=image_embed_interleave,
            )
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)
            negative_prompt_attention_mask = negative_prompt_attention_mask.to(transformer_dtype)
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(transformer_dtype)

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 0.0, num_inference_steps + 1)[:-1] if sigmas is None else sigmas
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, sigmas=sigmas)

        # 6. Prepare guidance condition
        guidance = None
        if self.transformer.config.guidance_embeds:
            guidance = (
                torch.tensor([guidance_scale] * latents.shape[0], dtype=transformer_dtype, device=device) * 1000.0
            )

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                if do_true_cfg and use_low_pass_guidance:
                    lp_strength = lp_utils.get_lp_strength(
                        step_index=i,
                        total_steps=num_inference_steps,
                        lp_strength_schedule_type=lp_strength_schedule_type,
                        schedule_interval_start_time=schedule_interval_start_time,
                        schedule_interval_end_time=schedule_interval_end_time,
                        schedule_linear_start_weight=schedule_linear_start_weight,
                        schedule_linear_end_weight=schedule_linear_end_weight,
                        schedule_linear_end_time=schedule_linear_end_time,
                        schedule_exp_decay_rate=schedule_exp_decay_rate,
                    )

                    modulated_lp_blur_sigma = lp_blur_sigma * lp_strength
                    if schedule_blur_kernel_size:
                        modulated_lp_blur_kernel_size = lp_blur_kernel_size * lp_strength
                    else:
                        modulated_lp_blur_kernel_size = lp_blur_kernel_size

                    # No-effect resize_factor is 1.0
                    modulated_lp_resize_factor = 1.0 - (1.0 - lp_resize_factor) * lp_strength

                    if enable_lp_img_embeds:
                        assert False, "Low-pass filter on image embeds is not supported in HunyuanVideo pipeline. Please set enable_lp_img_embeds = False"

                    lp_image_latents = self.prepare_lp(
                        lp_filter_type=lp_filter_type,
                        lp_blur_sigma=modulated_lp_blur_sigma,
                        lp_blur_kernel_size=modulated_lp_blur_kernel_size,
                        lp_resize_factor=modulated_lp_resize_factor,
                        generator=generator,
                        num_frames=num_frames,
                        use_low_pass_guidance=use_low_pass_guidance,
                        lp_filter_in_latent=lp_filter_in_latent,
                        orig_image_latents=image_latents,
                        orig_image_tensor=image 
                    )
                    if lp_strength == 0.0 or lp_on_noisy_latent:
                        latent_model_input = torch.cat([latents] * 2)
                        img_cond = torch.cat([image_latents,image_latents], dim=0).to(transformer_dtype)
                        latent_model_input = torch.cat([img_cond, latent_model_input[:, :, 1:]], dim=2).to(transformer_dtype)

                        concat_prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                        concat_pooled_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
                        concat_prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)
                    else:
                        latent_model_input = torch.cat([latents] * 3)
                        img_cond = torch.cat([image_latents,lp_image_latents,lp_image_latents], dim=0)
                        latent_model_input = torch.cat([img_cond, latent_model_input[:, :, 1:]], dim=2).to(transformer_dtype)
                        concat_prompt_embeds = torch.cat([negative_prompt_embeds,negative_prompt_embeds, prompt_embeds], dim=0)
                        concat_pooled_embeds = torch.cat([negative_pooled_prompt_embeds,negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
                        concat_prompt_attention_mask = torch.cat([negative_prompt_attention_mask,negative_prompt_attention_mask, prompt_attention_mask], dim=0)                        
                elif do_true_cfg:
                    latent_model_input = torch.cat([latents] * 2)
                    img_cond = torch.cat([image_latents,image_latents], dim=0).to(transformer_dtype)
                    latent_model_input = torch.cat([img_cond, latent_model_input[:, :, 1:]], dim=2).to(transformer_dtype)

                    concat_prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                    concat_pooled_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
                    concat_prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)
                elif not use_low_pass_guidance:
                    latent_model_input = torch.cat([image_latents, latents[:, :, 1:]], dim=2).to(transformer_dtype)
                    concat_prompt_embeds =  prompt_embeds
                    concat_pooled_embeds = pooled_prompt_embeds
                    concat_prompt_attention_mask = prompt_attention_mask
                else:
                    lp_strength = lp_utils.get_lp_strength(
                        step_index=i,
                        total_steps=num_inference_steps,
                        lp_strength_schedule_type=lp_strength_schedule_type,
                        schedule_interval_start_time=schedule_interval_start_time,
                        schedule_interval_end_time=schedule_interval_end_time,
                        schedule_linear_start_weight=schedule_linear_start_weight,
                        schedule_linear_end_weight=schedule_linear_end_weight,
                        schedule_linear_end_time=schedule_linear_end_time,
                        schedule_exp_decay_rate=schedule_exp_decay_rate,
                    )

                    modulated_lp_blur_sigma = lp_blur_sigma * lp_strength
                    if schedule_blur_kernel_size:
                        modulated_lp_blur_kernel_size = lp_blur_kernel_size * lp_strength
                    else:
                        modulated_lp_blur_kernel_size = lp_blur_kernel_size

                    modulated_lp_resize_factor = 1.0 - (1.0 - lp_resize_factor) * lp_strength

                    if enable_lp_img_embeds:
                        assert False, "Low-pass filter on image embeds is not supported in HunyuanVideo pipeline. Please set enable_lp_img_embeds = False"

                    lp_image_latents = self.prepare_lp(
                        lp_filter_type=lp_filter_type,
                        lp_blur_sigma=modulated_lp_blur_sigma,
                        lp_blur_kernel_size=modulated_lp_blur_kernel_size,
                        lp_resize_factor=modulated_lp_resize_factor,
                        generator=generator,
                        num_frames=num_frames,
                        use_low_pass_guidance=use_low_pass_guidance,
                        lp_filter_in_latent=lp_filter_in_latent,
                        orig_image_latents=image_latents,
                        orig_image_tensor=image 
                    )
                    latent_model_input = torch.cat([lp_image_latents, latents[:, :, 1:]], dim=2).to(transformer_dtype)
                    concat_prompt_embeds =  prompt_embeds
                    concat_pooled_embeds = pooled_prompt_embeds
                    concat_prompt_attention_mask = prompt_attention_mask

                timestep = t.expand(latent_model_input.shape[0]).to(transformer_dtype)
                latent_model_input = latent_model_input.to(transformer_dtype)
                prompt_embeds = prompt_embeds.to(transformer_dtype)
                prompt_attention_mask = prompt_attention_mask.to(transformer_dtype)
                pooled_prompt_embeds = pooled_prompt_embeds.to(transformer_dtype)

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=concat_prompt_embeds,
                    encoder_attention_mask=concat_prompt_attention_mask,
                    pooled_projections=concat_pooled_embeds,
                    guidance=guidance,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]

                if noise_pred.shape[0] == 3:
                    noise_pred_uncond_init, noise_pred_uncond, noise_pred_text = noise_pred.chunk(3)
                    noise_pred = (
                        noise_pred_uncond_init + true_cfg_scale * (noise_pred_text - noise_pred_uncond)
                    )
                elif noise_pred.shape[0] == 2:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + true_cfg_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                if image_condition_type == "latent_concat":
                    latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                elif image_condition_type == "token_replace":
                    latents = latents = self.scheduler.step(
                        noise_pred[:, :, 1:], t, latents[:, :, 1:], return_dict=False
                    )[0]
                    latents = torch.cat([image_latents, latents], dim=2)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        if not output_type == "latent":
            latents = latents.to(self.vae.dtype) / self.vae_scaling_factor
            video = self.vae.decode(latents, return_dict=False)[0]
            if image_condition_type == "latent_concat":
                video = video[:, :, 4:, :, :]
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            if image_condition_type == "latent_concat":
                video = latents[:, :, 1:, :, :]
            else:
                video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return HunyuanVideoPipelineOutput(frames=video)
