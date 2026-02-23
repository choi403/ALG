# Improving Motion in Image-to-Video Models via Adaptive Low-Pass Guidance

[<u>`Project Page`</u>](https://choi403.github.io/ALG/) | [<u>`arXiv`</u>](https://arxiv.org/abs/2506.08456) | [<u>`Gallery`</u>](https://choi403.github.io/ALG/gallery/)

Official implementation for [<u><b>Improving Motion in Image-to-Video Models via Adaptive Low-Pass Guidance</b></u>](https://arxiv.org/abs/2506.08456)
<br>
<a href="https://choi403.github.io/"><u>June Suk Choi</u></a>,
<a href="https://kyungmnlee.github.io/"><u>Kyungmin Lee</u></a>,
<a href="https://sihyun.me"><u>Sihyun Yu</u></a>,
<a href="https://scholar.google.com/citations?user=pM4aZGYAAAAJ&hl=en"><u>Yisol Choi</u></a>,
<a href="https://alinlab.kaist.ac.kr/shin.html"><u>Jinwoo Shin</u></a>,
<a href="https://sites.google.com/view/kiminlee"><u>Kimin Lee</u></a>

https://github.com/user-attachments/assets/a1faada7-624a-4259-8b40-dcef50700346

**Summary**: We propose **Adaptive Low-pass Guidance (ALG)**, a simple yet effective sampling method for pre-trained Image-to-Video (I2V) models. ALG mitigates the common issue of motion suppression by adaptively applying low-pass filtering to the conditioning image during the early stages of the denoising process. This encourages the generation of more dynamic videos without compromising the visual quality or fidelity to the input image.

## 1. Setup
```bash
conda create -n alg python=3.11 -y
conda activate alg
pip install -r requirements.txt # We recommend using torch version 2.5.1 and CUDA version 12.2 for the best compatibility.
```

## 2. How to Run

You can use the main script `run.py` to generate videos using our method. Configuration files are located in `./configs`.

### Basic Usage

You can generate a video using the following command with your image file and prompt.

```bash
python run.py \
  --config [PATH_TO_CONFIG_FILE] \
  --image_path [PATH_TO_INPUT_IMAGE] \
  --prompt "[YOUR_PROMPT]" \
  --output_path [PATH_TO_SAVE_VIDEO]
```

### Examples
We include a few example images in the asset folder, coupled with their corresponding prompts below.

**Generate a video with ALG enabled (more dynamic)**
```bash
python run.py \
  --config ./configs/wan_alg.yaml \
  --image_path ./assets/city.png \
  --prompt "A car chase through narrow city streets at night." \
  --output_path city_alg.mp4
```

**Generate a video without ALG (more static)**
```bash
python run.py \
  --config ./configs/wan_default.yaml \
  --image_path ./assets/city.png \
  --prompt "A car chase through narrow city streets at night." \
  --output_path city_baseline.mp4
```

**Example prompts**
```
city.png: "A car chase through narrow city streets at night."
snowboard.png: "A snowboarder doing a backflip off a jump."
boat.png: "A group of people whitewater rafting in a canyon."
helicopter.png: "A helicopter hovering over a rescue site."
tennis.png: "A man swinging a tennis racquet at a tennis ball."
```

## Configuration

All generation and ALG parameters are defined in a single yaml config file (e.g., `config/wan_alg.yaml`).

### Model configuration
```yaml
# configs/cogvideox_alg.yaml

model:
  path: "THUDM/CogVideoX-5b-I2V"   # Hugging Face model path
  dtype: "bfloat16"               # Dtype for the model (e.g., float16, bfloat16, float32)

generation:
  height: null                    # Output video height (null for model default)
  width: null                     # Output video width (null for model default)
  num_frames: 49                  # Number of frames to generate
  num_inference_steps: 50         # Denoising steps
  guidance_scale: 6.0             # Classifier-Free Guidance scale

video:
  fps: 12                         # FPS for the output video file
```

### ALG configuration (low-pass filtering)
*   `use_low_pass_guidance` (`bool`): Enable (`true`) or disable ALG for inference.

*   **Filter Settings**: Low-pass filtering characteristics.

    *   `lp_filter_type` (`str`): Specifies the type of low-pass filter to use.
        *   `"down_up"`: (Recommended) Bilinearly downsamples the image by `lp_resize_factor` and then upsamples it back to the original size.
        *   `"gaussian_blur"`: Applies Gaussian blur.

    *   `lp_filter_in_latent` (`bool`): Determines whether the filter is applied in pixel space or latent space.
        *   `true`: (Recommended) The filter is applied to the image's latent representation after it has been encoded by the VAE.
        *   `false`: The filter is applied directly to the RGB image *before* it is encoded by the VAE.

    *   `lp_resize_factor` (`float`): (for `"down_up"`)
        *   The factor by which to downsample the image (e.g., `0.25` means resizing to 25% of the original dimensions). Smaller value means stronget low-pass filtering, and potentially more motion.

    *   `lp_blur_sigma` (`float`): (for `"gaussian_blur"`)
        *   The standard deviation (sigma) for the Gaussian kernel. Larger values result in a stronger blur.

    *   `lp_blur_kernel_size` (`float` | `int`): (for `"gaussian_blur"`)
        *   The size of the blurring kernel. If a float, it's interpreted as a fraction of the image height.

*   **Adaptive Scheduling**: Controls how the strength of the low-pass filter changes over the denoising timesteps.

    *   `lp_strength_schedule_type` (`str`): The scheduling strategy. Strength is a multiplier from 0.0 (off) to 1.0 (full).
        *   `"interval"`: (Recommended) Applies the filter at full strength (`1.0`) for a specified portion of the denoising process and turns it off (`0.0`) for the rest.
        *   `"linear"`: Linearly decays the filter strength from a starting value to an ending value.
        *   `"exponential"`: Exponentially decays the filter strength from the beginning.
        *   `"none"`: Applies filter at a constant strength throughout.

    *   Parameters for `"interval"` schedule:
        *   `schedule_interval_start_time` (`float`): The point to turn the filter on, as a fraction of total steps [`0.0`,`1.0`]. `0.0` is the first step.
        *   `schedule_interval_end_time` (`float`): The point to turn the filter off. With 50 steps, `0.06` means the filter is active for the first `50 * 0.06 = 3` steps.

    *   Parameters for `"linear"` schedule:
        *   `schedule_linear_start_weight` (`float`): The filter strength at the first timestep (usually `1.0`).
        *   `schedule_linear_end_weight` (`float`): The final filter strength to decay towards (usually `0.0`).
        *   `schedule_linear_end_time` (`float`): The point in the process (as a fraction of total steps) at which the `end_weight` is reached. The strength remains at `end_weight` after this point.

    *   Parameters for `"exponential"` schedule:
        *   `schedule_exp_decay_rate` (`float`): The decay rate `r` for the formula `strength = exp(-r * time_fraction)`. Higher values cause strength to decay more quickly.

    *   `schedule_blur_kernel_size` (`bool`): If `true` and using a scheduler with the `"gaussian_blur"` filter, the blur kernel size will also be scaled down along with the filter strength.

## 3. Supported Models

We provide implementations and configurations for the following models:

*   **[CogVideoX](https://huggingface.co/THUDM/CogVideoX-5b-I2V)**: `THUDM/CogVideoX-5b-I2V`
*   **[Wan 2.1](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P-Diffusers)**: `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers`
*   **[HunyuanVideo](https://huggingface.co/tencent/HunyuanVideo-I2V)**: `tencent/HunyuanVideo-I2V`
*   [LTX-Video](https://huggingface.co/Lightricks/LTX-Video): `Lightricks/LTX-Video` (Not available yet, coming soon!)

We plan to add ALG implementation for LTX-Video as soon as possible!

You can create new configuration files for these models by modifying the `model.path` and adjusting the `generation` and `alg` parameters accordingly. Example configs are provided in the `./configs` directory.

## 4. More Examples

For more qualitative results and video comparisons, please visit the **[Gallery](https://choi403.github.io/ALG/gallery/)** on our project page.

## Acknowledgement

This code is built upon [Hugging Face Diffusers](https://github.com/huggingface/diffusers) library. We thank the authors of the open-source Image-to-Video models used in our work for making their code and models publicly available.

## BibTeX

If you find our work useful for your research, please consider citing our paper:

```bibtex
@article{choi2025alg,
  title={Improving Motion in Image-to-Video Models via Adaptive Low-Pass Guidance},
  author={Choi, June Suk and Lee, Kyungmin and Yu, Sihyun and Choi, Yisol and Shin, Jinwoo and Lee, Kimin},
  year={2025},
  journal={arXiv preprint arXiv:2506.08456},
}
```
