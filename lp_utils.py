import math
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvF
import numpy as np


def apply_low_pass_filter(
    tensor: torch.Tensor,
    filter_type: str,
    # Gaussian Blur Params
    blur_sigma: float,
    blur_kernel_size: float,  # Can be float (relative) or int (absolute)
    # Down/Up Sampling Params
    resize_factor: float,
):
    """
    Applies the specified low-pass filtering operation to the input tensor.
    Handles 4D ([B, C, H, W]) and 5D ([B, C, F, H, W]) tensors by temporarily
    reshaping 5D tensors for spatial filtering.
    """
    # --- Early Exits for No-Op Cases ---
    if filter_type == "none":
        return tensor
    if filter_type == "down_up" and resize_factor == 1.0:
        return tensor
    if filter_type == "gaussian_blur" and blur_sigma == 0:
        return tensor

    # --- Reshape 5D tensor for spatial filtering ---
    is_5d = tensor.ndim == 5
    if is_5d:
        B, C, K, H, W = tensor.shape
        # Flatten frames into batch dimension using view
        tensor = tensor.view(B * K, C, H, W)
    else:
        B, C, H, W = tensor.shape

    # --- Apply Selected Filter ---
    if filter_type == "gaussian_blur":
        if isinstance(blur_kernel_size, float):
            kernel_val = max(int(blur_kernel_size * H), 1)
        else:
            kernel_val = int(blur_kernel_size)
        if kernel_val % 2 == 0:
            kernel_val += 1
        tensor = tvF.gaussian_blur(tensor, kernel_size=[kernel_val, kernel_val], sigma=[blur_sigma, blur_sigma])

    elif filter_type == "down_up":
        h0, w0 = tensor.shape[-2:]
        h1 = max(1, int(round(h0 * resize_factor)))
        w1 = max(1, int(round(w0 * resize_factor)))
        tensor = F.interpolate(tensor, size=(h1, w1), mode="bilinear", align_corners=False, antialias=True)
        tensor = F.interpolate(tensor, size=(h0, w0), mode="bilinear", align_corners=False, antialias=True)

    # --- Restore original 5D shape if necessary ---
    if is_5d:
        tensor = tensor.view(B, C, K, H, W)

    return tensor


def get_lp_strength(
    step_index: int,
    total_steps: int,
    lp_strength_schedule_type: str,
    # Interval params
    schedule_interval_start_time: float,
    schedule_interval_end_time: float,
    # Linear params
    schedule_linear_start_weight: float,
    schedule_linear_end_weight: float,
    schedule_linear_end_time: float,
    # Exponential params
    schedule_exp_decay_rate: float,
) -> float:
    """
    Calculates the low-pass guidance strength multiplier for the current timestep
    based on the specified schedule.
    """
    step_norm = step_index / max(total_steps - 1, 1)

    if lp_strength_schedule_type == "linear":
        schedule_duration_fraction = schedule_linear_end_time
        if schedule_duration_fraction <= 0:
            return schedule_linear_start_weight
        if step_norm >= schedule_duration_fraction:
            current_strength = schedule_linear_end_weight
        else:
            progress = step_norm / schedule_duration_fraction
            current_strength = schedule_linear_start_weight * (1 - progress) + schedule_linear_end_weight * progress
        return current_strength

    elif lp_strength_schedule_type == "interval":
        if schedule_interval_start_time <= step_norm <= schedule_interval_end_time:
            return 1.0
        else:
            return 0.0

    elif lp_strength_schedule_type == "exponential":
        decay_rate = schedule_exp_decay_rate
        if decay_rate < 0:
            print(f"Warning: Negative exponential_decay_rate ({decay_rate}) is unusual. Using abs value.")
            decay_rate = abs(decay_rate)
        return math.exp(-decay_rate * step_norm)

    elif lp_strength_schedule_type == "none":
        return 1.0
    else:
        print(f"Warning: Unknown lp_strength_schedule_type '{lp_strength_schedule_type}'. Using constant strength 1.0.")
        return 1.0

def _generate_crop_size_list(base_size=256, patch_size=32, max_ratio=4.0):
    """generate crop size list (HunyuanVideo)

    Args:
        base_size (int, optional): the base size for generate bucket. Defaults to 256.
        patch_size (int, optional): the stride to generate bucket. Defaults to 32.
        max_ratio (float, optional): th max ratio for h or w based on base_size . Defaults to 4.0.

    Returns:
        list: generate crop size list
    """
    num_patches = round((base_size / patch_size) ** 2)
    assert max_ratio >= 1.0
    crop_size_list = []
    wp, hp = num_patches, 1
    while wp > 0:
        if max(wp, hp) / min(wp, hp) <= max_ratio:
            crop_size_list.append((wp * patch_size, hp * patch_size))
        if (hp + 1) * wp <= num_patches:
            hp += 1
        else:
            wp -= 1
    return crop_size_list

def _get_closest_ratio(height: float, width: float, ratios: list, buckets: list):
    """get the closest ratio in the buckets (HunyuanVideo)

    Args:
        height (float): video height
        width (float): video width
        ratios (list): video aspect ratio
        buckets (list): buckets generate by `generate_crop_size_list`

    Returns:
        the closest ratio in the buckets and the corresponding ratio
    """
    aspect_ratio = float(height) / float(width)
    diff_ratios = ratios - aspect_ratio

    if aspect_ratio >= 1:
        indices = [(index, x) for index, x in enumerate(diff_ratios) if x <= 0]
    else:
        indices = [(index, x) for index, x in enumerate(diff_ratios) if x > 0]

    closest_ratio_id = min(indices, key=lambda pair: abs(pair[1]))[0]
    closest_size = buckets[closest_ratio_id]
    closest_ratio = ratios[closest_ratio_id]

    return closest_size, closest_ratio

def get_hunyuan_video_size(i2v_resolution, input_image):
    """
    Map to target height and width based on resolution for HunyuanVideo

    Args:
        height (float): video height
        width (float): video width
        ratios (list): video aspect ratio
        buckets (list): buckets generate by `generate_crop_size_list`

    Returns:
        the closest ratio in the buckets and the corresponding ratio
    """
    if i2v_resolution == "720p":
        bucket_hw_base_size = 960
    elif i2v_resolution == "540p":
        bucket_hw_base_size = 720
    elif i2v_resolution == "360p":
        bucket_hw_base_size = 480

    origin_size = input_image.size

    crop_size_list = _generate_crop_size_list(bucket_hw_base_size, 32)
    aspect_ratios = np.array([round(float(h)/float(w), 5) for h, w in crop_size_list])
    closest_size, _ = _get_closest_ratio(origin_size[1], origin_size[0], aspect_ratios, crop_size_list)
    target_height, target_width = closest_size
    return target_height, target_width