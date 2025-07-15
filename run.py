import yaml
import argparse
import torch
import torchvision
from PIL import Image
import logging
import sys

# --- Diffusers and Transformers Imports ---
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, HunyuanVideoTransformer3DModel, FlowMatchEulerDiscreteScheduler
from diffusers.utils import load_image
from transformers import CLIPVisionModel

# --- Low-pass Pipelines ---
from pipeline_wan_image2video_lowpass import WanImageToVideoPipeline
from pipeline_cogvideox_image2video_lowpass import CogVideoXImageToVideoPipeline
from pipeline_hunyuan_video_image2video_lowpass import HunyuanVideoImageToVideoPipeline

from lp_utils import get_hunyuan_video_size

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)


def main(args):
    # 1. Configuration
    IMAGE_PATH = args.image_path
    PROMPT = args.prompt
    OUTPUT_PATH = args.output_path
    MODEL_CACHE_DIR = args.model_cache_dir

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    model_path = config['model']['path']
    model_dtype_str = config['model']['dtype']
    model_dtype = getattr(torch, model_dtype_str)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Using device: {device}")

    # 2. Pipeline preparation
    if "Wan" in model_path:
        image_encoder = CLIPVisionModel.from_pretrained(model_path,
            subfolder="image_encoder",
            torch_dtype=torch.float32,
            cache_dir=MODEL_CACHE_DIR
        )
        vae = AutoencoderKLWan.from_pretrained(model_path,
            subfolder="vae",
            torch_dtype=torch.float32,
            cache_dir=MODEL_CACHE_DIR
        )
        pipe = WanImageToVideoPipeline.from_pretrained(model_path,
            vae=vae,
            image_encoder=image_encoder,
            torch_dtype=model_dtype,
            cache_dir=MODEL_CACHE_DIR
        )
        # Recommended setup (See https://github.com/huggingface/diffusers/blob/3c8b67b3711b668a6e7867e08b54280e51454eb5/src/diffusers/pipelines/wan/pipeline_wan.py#L58C13-L58C23)
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0 if config['generation']['height'] == '480' else 5.0)
    elif "CogVideoX" in model_path:
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(
            model_path,
            torch_dtype=model_dtype,
            cache_dir=MODEL_CACHE_DIR
        )
    elif "HunyuanVideo" in model_path:
        transformer = HunyuanVideoTransformer3DModel.from_pretrained(
            model_path,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
            cache_dir=MODEL_CACHE_DIR
        )
        pipe = HunyuanVideoImageToVideoPipeline.from_pretrained(
            model_path, transformer=transformer,
            torch_dtype=torch.float16,
            cache_dir=MODEL_CACHE_DIR
        )
        pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            pipe.scheduler.config,
            flow_shift= config['model']['flow_shift'],
            invert_sigmas = config['model']['flow_reverse']
        )
    pipe.to(device)

    logger.info("Pipeline loaded successfully.")

    # 3. Prepare inputs
    input_image = load_image(Image.open(IMAGE_PATH))

    generator = torch.Generator(device=device).manual_seed(42)

    pipe_kwargs = {
        "image": input_image,
        "prompt": PROMPT,
        "generator": generator,
    }

    params_from_config = {**config.get('generation', {}), **config.get('alg', {})}

    for key, value in params_from_config.items():
        if value is not None:
            pipe_kwargs[key] = value

    logger.info("Starting video generation...")
    log_subset = {k: v for k, v in pipe_kwargs.items() if k not in ['image', 'generator']}
    logger.info(f"Pipeline arguments: {log_subset}")

    if "HunyuanVideo" in model_path:
        pipe_kwargs["height"], pipe_kwargs["width"] = get_hunyuan_video_size(config['video']['resolution'], input_image)

    # 4. Generate video
    video_output = pipe(**pipe_kwargs)
    video_frames = video_output.frames[0]  # Output is a list containing a list of PIL Images
    logger.info(f"Video generation complete. Received {len(video_frames)} frames.")

    # 5. Save video
    video_tensors = [torchvision.transforms.functional.to_tensor(frame) for frame in video_frames]
    video_tensor = torch.stack(video_tensors)  # Shape: (T, C, H, W)
    video_tensor = video_tensor.permute(0, 2, 3, 1)  # Shape: (T, H, W, C) for write_video
    video_tensor = (video_tensor * 255).clamp(0, 255).to(torch.uint8).cpu()

    logger.info(f"Saving video to: {OUTPUT_PATH}")
    torchvision.io.write_video(
        OUTPUT_PATH,
        video_tensor,
        fps=config['video']['fps'],
        video_codec='h264',
        options={'crf': '18', 'preset': 'slow'}
    )
    logger.info("Video saved successfully. Run complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("--config", type=str, default="./configs/hunyuan_video_alg.yaml")
    parser.add_argument("--image_path", type=str, default="./assets/a red double decker bus driving down a street.jpg")
    parser.add_argument("--prompt", type=str, default="a red double decker bus driving down a street")
    parser.add_argument("--output_path", type=str, default="output.mp4")
    parser.add_argument("--model_cache_dir", type=str, default=None)
    args = parser.parse_args()

    main(args)