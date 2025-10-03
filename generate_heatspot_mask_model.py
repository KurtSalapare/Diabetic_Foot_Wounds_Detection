# --------------------------------------------
# Wound synthesis with:
# - Foot correction (mirror/rotate/scale/pad) before any wound logic
# - Contralateral-referenced temperature evolution (0 -> 3.5°C over DEV_DAYS)
# - Relative wound sizing (with region coverage caps)
# - Randomized side/region/position; synthetic timeline via modulo of source days
# - Concise per-variant terminal update
# --------------------------------------------

import os
import warnings
import numpy as np
import random
import scipy.io
import matplotlib.pyplot as plt
from scipy.io.matlab import MatReadWarning
from scipy.ndimage import gaussian_filter
from skimage.transform import rescale

from foot_part_identifier import segment_foot
# OLD
# from foot_overlay_creator import (
#     to_nan, trim_to_content, mirror_horiz, pad_to_same_size,
#     rotate_image_preserve_temps, find_best_rotation_angle,
#     scale_image_preserve_temps
# )

# NEW
from foot_overlay_creator_simplified2 import (
    to_nan, trim_to_content, mirror_horiz, pad_to_same_size,
    rotate_image, find_best_rotation_angle, scale_image, create_binary_mask
)
import torch


warnings.filterwarnings("ignore", category=MatReadWarning)

# Silence noisy prints from imported helpers when enabled
SUPPRESS_IMPORTED_LOGS = True

from contextlib import contextmanager, redirect_stdout, redirect_stderr
import sys, io, os


@contextmanager
def _silence_imported(enabled=True):
    if not enabled:
        yield
        return
    with open(os.devnull, "w") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            yield


# ==========================
# HIGH-LEVEL DATASET CONFIG
# ==========================
RUN_ALL_PATIENTS = False
SINGLE_PATIENT_ID = "gz15"
PATIENT_PREFIX = "gz"
PATIENT_COUNT = 15
MAT_ROOT = "Data/Temp Data"

WOUND_VARIANTS_PER_PATIENT = 3
OUTPUT_ROOT = "output_images_wound_modes_segmented"

GLOBAL_SEED = None  # e.g., 123

# Log one concise line per variant
LOG_VARIANT_ASSIGNMENTS = True

# ==========================
# GENERATION/GROWTH CONFIG
# ==========================
GENERATION_MODE = "both"  # "static", "developing", "both"
DEV_DAYS = 20
STATIC_DAYS = 10
PRE_WOUND_DAYS = 10

DEVELOP_MODE = "size+intensity"  # "size+intensity" | "intensity-only"
INITIAL_SIZE_SCALE = 0.05
INITIAL_TEMP_SCALE = 0.05  # kept for compatibility (not used by new temp evolution)

# ---- Relative size policy (prevents huge wounds on padded canvases) ----
SIZE_POLICY = "relative"  # "relative" (recommended) | "absolute"

# Relative sizing (fractions of min(ref_h, ref_w))
CORE_RADIUS_FRAC_RANGE = (0.025, 0.040)  # ~2.5%..4.0% of min dimension
INFLAM_OVER_CORE_RATIO = (1.6, 2.2)  # inflam radius = core * ratio
# Max final coverage (fraction of target region)
MAX_INFLAM_REGION_COVERAGE = 0.08  # 8%
MAX_CORE_REGION_COVERAGE = 0.03  # 3%

# Absolute sizing fallback (narrowed)
ABS_CORE_RADIUS_RANGE = (8, 15)
ABS_INFLAM_RADIUS_RANGE = (16, 28)

# Visualization / formatting
FIGSIZE = (9, 6)
CMAP = "hot"
GAP_COLS = 5
CBAR_FRACTION = 0.035
CBAR_PAD = 0.04

# Temperature increment target
FINAL_INCREMENT_DEG_C = 3.0  # progresses 0 -> 3.0 °C over DEV_DAYS

# Debug mark (Day 1 only)
MARK_CENTER_ON_FIRST_DAY = True
CENTER_DOT_SIZE = 36  # tweak if you want it bigger/smaller


# --------------------------------------------
# Randomization helpers (per-variant diversity)
# --------------------------------------------
def rng(seed=None):
    return np.random.default_rng(seed) if seed is not None else np.random.default_rng()


def sample_variant_params(rng_):
    # Radii are chosen later (relative policy) after we know canvas/region.
    params = {
        "apply_to": rng_.choice(["left", "right"]),
        "apply_wound_to": rng_.choice(["heel", "upper_foot", "mid_foot"]),
        "position_mode": 2,  # random in region
        "manual_coord": (150, 180),
        "shape_mode": rng_.choice(["circle", "multi"]),
        # Legacy placeholders; used only if SIZE_POLICY="absolute"
        "core_radius_final": int(rng_.integers(*ABS_CORE_RADIUS_RANGE)),
        "inflam_radius_final": int(rng_.integers(*ABS_INFLAM_RADIUS_RANGE)),
        "blur_sigma_core": float(rng_.uniform(5.0, 7.0)),
        "blur_sigma_inflam": float(rng_.uniform(5.0, 7.0)),
        "multi_min_blobs": 2,
        "multi_max_blobs": 6,
        "develop_mode": DEVELOP_MODE,
        "initial_size_scale": INITIAL_SIZE_SCALE,
        "initial_temp_scale": INITIAL_TEMP_SCALE,
    }
    return params


# ==========================
# Helpers
# ==========================

def generate_weighted_random(x, y, z):
    # Generate a random float between 0.0 and 1.0
    probability_roll = random.random()
    
    # 60% chance (if probability_roll is between 0.0 and 0.6)
    if probability_roll < 0.60:
        return x
    else:
        # 40% chance (if probability_roll is between 0.6 and 1.0)
        # Generate a uniform random integer between 10 and 30 (inclusive)
        return random.randint(y, z)

def to_display(img2d):
    return np.where(img2d == 0, np.nan, img2d)


def _build_full_region_mask(heel, mid, upper, region_key):
    """Return a full-height mask for exactly one of: 'heel', 'mid_foot', 'upper_foot'."""
    nan_heel = np.full_like(heel, np.nan)
    nan_mid = np.full_like(mid, np.nan)
    nan_up = np.full_like(upper, np.nan)

    if region_key == "heel":
        return np.vstack((heel, nan_mid, nan_up))
    elif region_key == "mid_foot":
        return np.vstack((nan_heel, mid, nan_up))
    elif region_key == "upper_foot":
        return np.vstack((nan_heel, nan_mid, upper))
    else:
        raise ValueError(
            f"apply_wound_to must be one of 'heel', 'mid_foot', 'upper_foot' (got: {region_key!r})"
        )


def select_center_and_shape_on_image(target_foot_img, apply_wound_to, position_mode, manual_coord, shape_mode, rng_):
    heel, mid_foot, upper_foot = segment_foot(target_foot_img)

    h, w = target_foot_img.shape

    region_key = apply_wound_to  # use exactly what caller provided
    region_full = _build_full_region_mask(heel, mid_foot, upper_foot, region_key)

    ys, xs = np.where(~np.isnan(region_full))
    if len(xs) == 0:
        # Fall back to any valid pixel on the target foot image (rare)
        ys, xs = np.where(~np.isnan(target_foot_img))

    if position_mode == 1:
        y_center = int(np.mean(ys));
        x_center = int(np.mean(xs))
    elif position_mode == 2:
        idx = np.random.randint(len(ys))
        y_center, x_center = ys[idx], xs[idx]
    elif position_mode == 3:
        y_center, x_center = manual_coord
    else:
        raise ValueError("Invalid position_mode")

    chosen_shape = shape_mode if shape_mode in ["circle", "multi"] else rng_.choice(["circle", "multi"])
    return y_center, x_center, chosen_shape, h, w, region_full, region_key


def make_circle_mask(cx, cy, radius, X, Y):
    """Helper to create a single circle mask using the 1D ogrid slices X and Y."""
    return (X - cx) ** 2 + (Y - cy) ** 2 <= radius ** 2


def build_final_mask(shape_mode, x_center, y_center, core_radius_final, inflam_radius_final,
                     multi_min_blobs, multi_max_blobs, h, w):
    """
    Returns: final_core_mask, final_inflam_mask, blob_count
    - blob_count = 1 for 'circle', or the sampled number of blobs for 'multi'
    """
    Y, X = np.ogrid[:h, :w]
    final_core_mask, final_inflam_mask, n_blobs = 0, 0, -1

    if shape_mode == "circle":
        final_core_mask = (X - x_center) ** 2 + (Y - y_center) ** 2 <= core_radius_final ** 2
        final_inflam_mask = (X - x_center) ** 2 + (Y - y_center) ** 2 <= inflam_radius_final ** 2
        n_blobs = 1

    # multi-blob union
    # n_blobs = np.random.randint(multi_min_blobs, multi_max_blobs + 1)
    else:
        # Configuration for the constraint
        n_blobs = np.random.randint(multi_min_blobs, multi_max_blobs + 1)
        MIN_UNIQUE_PIXELS = 10
        MAX_ATTEMPTS = 500  # Increased attempts for better chance of finding a spot

        # Lists to store parameters of successfully placed blobs
        centers = []
        core_r_list = []
        inflam_r_list = []

        # List of individual core masks (used to calculate cumulative mask)
        individual_core_masks = []

        # --- 1. Place the first blob (Unconstrained) ---
        initial_core_mask = make_circle_mask(x_center, y_center, core_radius_final, X, Y)

        individual_core_masks.append(initial_core_mask)
        centers.append((x_center, y_center))
        core_r_list.append(core_radius_final)
        inflam_r_list.append(inflam_radius_final)

        # --- 2. Iterative Placement with Constraint Check ---

        for i in range(n_blobs - 1):

            # The cumulative mask is the union of all successfully placed core blobs so far
            cumulative_core_mask = np.logical_or.reduce(individual_core_masks)

            blob_placed = False
            for attempt in range(MAX_ATTEMPTS):

                # a. Generate new blob parameters (Position and Radius)
                # Base the new center on a randomly chosen existing center (clustering)
                base_x, base_y = random.choice(centers)

                angle = np.random.uniform(0, 2 * np.pi)

                # Determine distance relative to the core radius
                min_dist = max(2, core_radius_final // 2)
                max_dist = max(3, core_radius_final * 2)
                dist = np.random.randint(min_dist, max_dist)

                dx, dy = int(np.cos(angle) * dist), int(np.sin(angle) * dist)

                # New center, clipped to stay within image boundaries
                new_x = np.clip(base_x + dx, 0, w - 1)
                new_y = np.clip(base_y + dy, 0, h - 1)

                # New radii (randomly generated)
                new_rc = np.random.randint(max(2, core_radius_final // 2), max(3, core_radius_final))
                new_ri = np.random.randint(max(2, inflam_radius_final // 2), max(3, inflam_radius_final))

                # b. Create the mask for the potential new blob
                new_core_mask = make_circle_mask(new_x, new_y, new_rc, X, Y)

                # c. Find the unique area: (New Mask) AND (NOT Cumulative Mask)
                unique_pixels_mask = new_core_mask & (~cumulative_core_mask)
                unique_count = np.sum(unique_pixels_mask)

                # d. Check the constraint (unique pixels >= 5)
                if unique_count >= MIN_UNIQUE_PIXELS:
                    # Constraint met: Add this validated blob and move to the next iteration
                    individual_core_masks.append(new_core_mask)
                    centers.append((new_x, new_y))
                    core_r_list.append(new_rc)
                    inflam_r_list.append(new_ri)
                    blob_placed = True
                    break

            if not blob_placed:
                print(
                    f"Warning: Failed to place blob {i + 2} after {MAX_ATTEMPTS} attempts. Constraint may be too strict.")

        # --- 3. Final Merging ---

        # The core mask is the union of all validated individual masks
        final_core_mask = np.logical_or.reduce(individual_core_masks)

        # The inflammation mask is created using all validated centers/radii
        # (It does NOT need the overlap check, it just uses the final parameters)
        final_inflam_mask = np.zeros((h, w), dtype=bool)
        for (cx, cy), ri in zip(centers, inflam_r_list):
            final_inflam_mask |= make_circle_mask(cx, cy, ri, X, Y)

        n_blobs = len(centers)

    return final_core_mask, final_inflam_mask, n_blobs



def scale_mask(mask, scale_factor, h, w):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return mask
    cx, cy = np.mean(xs), np.mean(ys)
    shift_x, shift_y = w // 2 - cx, h // 2 - cy
    shifted = np.roll(mask, (int(shift_y), int(shift_x)), axis=(0, 1)).astype(float)
    resized = rescale(shifted, scale=scale_factor, preserve_range=True, anti_aliasing=True, order=1)
    rh, rw = resized.shape
    out = np.zeros((h, w), dtype=bool)
    sy = max((h - rh) // 2, 0);
    sx = max((w - rw) // 2, 0)
    ey = min(h, sy + rh);
    ex = min(w, sx + rw)
    out[sy:ey, sx:ex] = resized[:ey - sy, :ex - sx] > 0.5
    final = np.roll(out, (-int(shift_y), -int(shift_x)), axis=(0, 1))
    return final


def masks_for_progress(progress, final_core_mask, final_inflam_mask, params, h, w):
    if params["develop_mode"] == "size+intensity":
        if progress < 1:
            scale_factor = (params["initial_size_scale"] + (1.0 - params["initial_size_scale"]) * progress)
            core_mask = scale_mask(final_core_mask, scale_factor, h, w)
            inflam_mask = scale_mask(final_inflam_mask, scale_factor, h, w)
        else :
            scale_factor = 1
            core_mask = scale_mask(final_core_mask, scale_factor, h, w)
            inflam_mask = scale_mask(final_inflam_mask, scale_factor, h, w)
    elif params["develop_mode"] == "intensity-only":
        core_mask = final_core_mask
        inflam_mask = final_inflam_mask
    else:
        raise ValueError("Unknown develop_mode")
    return core_mask, inflam_mask


def nanmean_safe(arr):
    vals = arr[~np.isnan(arr)]
    if vals.size == 0:
        return np.nan
    return float(np.mean(vals))


def soft_blend_set(base_canvas, target_value, mask_binary, sigma):
    if mask_binary.dtype != float:
        mask_float = mask_binary.astype(float)
    else:
        mask_float = mask_binary
    soft = gaussian_filter(mask_float, sigma=sigma)
    mmax = np.nanmax(soft)
    if not np.isfinite(mmax) or mmax == 0:
        return base_canvas
    soft = soft / mmax
    out = np.array(base_canvas, copy=True)
    valid = ~np.isnan(base_canvas)
    w = np.clip(soft, 0, 1)
    out[valid] = base_canvas[valid] * (1 - w[valid]) + target_value * w[valid]
    return out


def masked_nonzero_mean(canvas, mask):
    """
    Mean of values within 'mask' using only pixels that are both non-NaN and non-zero.
    If the mask has no such pixels, return 0.0 (no global fallback).
    """
    m = mask.astype(bool)
    if not np.any(m):
        return 0.0
    sub = canvas[m]
    valid = (~np.isnan(sub)) & (sub != 0)
    if np.any(valid):
        return float(np.mean(sub[valid]))
    return 0.0

def center_pad_to(img, target_h, target_w):
    h, w = img.shape
    canvas = np.full((target_h, target_w), np.nan, dtype=float)
    top = max(0, (target_h - h) // 2)
    left = max(0, (target_w - w) // 2)
    canvas[top:top + h, left:left + w] = img
    return canvas

# ---------------------------
# Patient-level foot correction (compute once) + per-day application
# ---------------------------

def _create_grid(h, w, device):
    y = torch.linspace(-1, 1, h, device=device)
    x = torch.linspace(-1, 1, w, device=device)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    return torch.stack([xx, yy], dim=-1).unsqueeze(0)  # [1, h, w, 2]

def _optimize_displacement(mask_left_np, mask_right_np, device, iters=150, lr=0.02, disp_scale=0.1):
    """Find coarse displacement that maximizes IoU between left and right masks (mirrored/rot+scaled already)."""
    mask_left_t  = torch.from_numpy(mask_left_np).float().to(device)
    mask_right_t = torch.from_numpy(mask_right_np).float().to(device)
    H, W = mask_left_t.shape

    identity_grid = _create_grid(H, W, device)
    # 3x3 coarse control points, bilinear-upsampled to full grid
    displ_coarse = torch.nn.Parameter(torch.zeros(1, 3, 3, 2, device=device))
    opt = torch.optim.Adam([displ_coarse], lr=lr)

    best_iou, best_displ = 0.0, None
    for _ in range(iters):
        opt.zero_grad()
        displ_full = torch.nn.functional.interpolate(
            displ_coarse.permute(0, 3, 1, 2), size=(H, W), mode='bilinear', align_corners=False
        ).permute(0, 2, 3, 1)
        grid = identity_grid + displ_full * disp_scale

        warped = torch.nn.functional.grid_sample(
            mask_right_t[None, None, ...], grid, mode='bilinear',
            padding_mode='zeros', align_corners=False
        ).squeeze()

        inter = torch.sum(warped * mask_left_t)
        union = torch.sum(warped + mask_left_t - warped * mask_left_t) + 1e-6
        iou = inter / union

        # simple regularization to keep displacement gentle/smooth
        disp_mag = torch.mean(torch.sum(displ_full ** 2, dim=-1))
        smooth = torch.mean(torch.abs(displ_full[:, 1:, :, :] - displ_full[:, :-1, :, :])) + \
                 torch.mean(torch.abs(displ_full[:, :, 1:, :] - displ_full[:, :, :-1, :]))

        loss = -iou + 0.1 * disp_mag + 0.05 * smooth
        loss.backward()
        opt.step()

        if iou.item() > best_iou:
            best_iou = iou.item()
            best_displ = displ_full.detach()

    # Final grid used everywhere for this patient
    final_grid = _create_grid(H, W, device) + best_displ * 0.1
    return final_grid  # [1,H,W,2] on device


def compute_patient_transform(left_crop, right_crop):
    """
    Compute mirror+rotation+scale+warp ONCE using day-0 scans.
    Returns:
      ref_left (H,W), ref_right (H,W), tf (dict with angle, scale, grid, ref_shape, fill_value)
    """
    # day-0 preprocessing
    left0  = trim_to_content(to_nan(left_crop[0, 0],  adaptive_threshold=True))
    right0 = trim_to_content(to_nan(right_crop[0, 0], adaptive_threshold=True))
    right0_m = mirror_horiz(right0)

    # rotation search (right mirrored vs left)
    angle, _score, _ = find_best_rotation_angle(left0, right0_m)
    if angle != 0:
        right0_m = trim_to_content(rotate_image(right0_m, angle))

    # scale right to left's size
    sx = left0.shape[1] / right0_m.shape[1] if right0_m.shape[1] else 1.0
    sy = left0.shape[0] / right0_m.shape[0] if right0_m.shape[0] else 1.0
    right0_s = scale_image(right0_m, sx, sy) if right0_m.shape != left0.shape else right0_m

    # build masks & optimize small warp to improve overlap (torch)
    mask_left  = create_binary_mask(left0)
    mask_right = create_binary_mask(right0_s)
    device = torch.device('cpu')
    grid = _optimize_displacement(mask_left.astype(float), mask_right.astype(float), device=device)

    # apply warp to temperature data (nearest to preserve temps)
    img_right_t = torch.from_numpy(right0_s).float().to(device)
    fill_value = float(np.nanmin(right0_s)) - 5.0 if not np.isnan(np.nanmin(right0_s)) else 15.0
    img_right_filled = torch.where(torch.isnan(img_right_t),
                                   torch.tensor(fill_value, dtype=torch.float32, device=device),
                                   img_right_t)
    valid_mask_t = (~torch.isnan(img_right_t)).float()

    warped_temp = torch.nn.functional.grid_sample(
        img_right_filled[None, None, ...], grid, mode='nearest',
        padding_mode='border', align_corners=False
    ).squeeze()
    warped_valid = torch.nn.functional.grid_sample(
        valid_mask_t[None, None, ...], grid, mode='nearest',
        padding_mode='zeros', align_corners=False
    ).squeeze()

    right0_warp = torch.where(warped_valid > 0.5, warped_temp,
                              torch.tensor(np.nan, dtype=torch.float32)).cpu().numpy()

    # pad to identical canvas (usually already equal)
    ref_left, ref_right = pad_to_same_size(left0, right0_warp)
    tf = {
        "angle": float(angle),
        "scale_x": float(sx),
        "scale_y": float(sy),
        "grid": grid,                  # torch tensor on CPU
        "ref_shape": ref_left.shape,   # (H,W)
        "fill_value": float(fill_value)
    }
    return ref_left, ref_right, tf


def apply_patient_transform(scan_left, scan_right, tf):
    """
    Apply the precomputed transform to any day.
    Outputs left_canvas, right_canvas, info (angle/scale/shape).
    """
    left  = trim_to_content(to_nan(scan_left,  adaptive_threshold=True))
    right = trim_to_content(to_nan(scan_right, adaptive_threshold=True))
    right_m = mirror_horiz(right)

    # rotate & scale with patient-level params
    if tf["angle"] != 0.0:
        right_m = trim_to_content(rotate_image(right_m, tf["angle"]))

    if right_m.shape != (int(round(right_m.shape[0] * tf["scale_y"])),
                         int(round(right_m.shape[1] * tf["scale_x"]))):
        right_s = scale_image(right_m, tf["scale_x"], tf["scale_y"])
    else:
        right_s = right_m

    # warp to reference frame
    device = torch.device('cpu')
    H, W = tf["ref_shape"]
    grid = tf["grid"]  # already sized [1,H,W,2]
    img_right_t = torch.from_numpy(right_s).float().to(device)
    img_right_filled = torch.where(torch.isnan(img_right_t),
                                   torch.tensor(tf["fill_value"], dtype=torch.float32, device=device),
                                   img_right_t)
    valid_mask_t = (~torch.isnan(img_right_t)).float()

    warped_temp = torch.nn.functional.grid_sample(
        img_right_filled[None, None, ...], grid, mode='nearest',
        padding_mode='border', align_corners=False
    ).squeeze()
    warped_valid = torch.nn.functional.grid_sample(
        valid_mask_t[None, None, ...], grid, mode='nearest',
        padding_mode='zeros', align_corners=False
    ).squeeze()
    right_w = torch.where(warped_valid > 0.5, warped_temp,
                          torch.tensor(np.nan, dtype=torch.float32)).cpu().numpy()

    # center-pad both to reference canvas
    left_can  = center_pad_to(left,  H, W) if left.shape  != (H, W) else left
    right_can = center_pad_to(right_w, H, W) if right_w.shape != (H, W) else right_w

    info = {
        "rotation_angle": tf["angle"],
        "scale_x": tf["scale_x"],
        "scale_y": tf["scale_y"],
        "canvas_shape": (H, W)
    }
    return left_can, right_can, info

# ---- Radii selection & caps ----
def pick_final_radii(h, w, region_full, rng_, params):
    if SIZE_POLICY == "relative":
        min_dim = float(min(h, w))
        core_frac = rng_.uniform(*CORE_RADIUS_FRAC_RANGE)
        core_r = int(max(2, round(min_dim * core_frac)))
        ratio = rng_.uniform(*INFLAM_OVER_CORE_RATIO)
        inflam_r = int(max(core_r + 1, round(core_r * ratio)))
    else:
        core_r = int(rng_.integers(*ABS_CORE_RADIUS_RANGE))
        inflam_r = int(rng_.integers(*ABS_INFLAM_RADIUS_RANGE))
        inflam_r = max(inflam_r, core_r + 1)
    return core_r, inflam_r


def enforce_region_coverage_cap(final_core_mask, final_inflam_mask, region_full, h, w):
    region_area = np.count_nonzero(~np.isnan(region_full))
    if region_area == 0:
        return final_core_mask, final_inflam_mask
    inflam_area = int(np.count_nonzero(final_inflam_mask))
    core_area = int(np.count_nonzero(final_core_mask))
    inflam_frac = inflam_area / region_area
    core_frac = core_area / region_area

    scale_needed = 1.0
    if inflam_frac > MAX_INFLAM_REGION_COVERAGE:
        scale_needed = min(scale_needed, np.sqrt(MAX_INFLAM_REGION_COVERAGE / max(inflam_frac, 1e-8)))
    if core_frac > MAX_CORE_REGION_COVERAGE:
        scale_needed = min(scale_needed, np.sqrt(MAX_CORE_REGION_COVERAGE / max(core_frac, 1e-8)))

    if scale_needed < 1.0:
        final_inflam_mask = scale_mask(final_inflam_mask, scale_needed, h, w)
        final_core_mask = scale_mask(final_core_mask, scale_needed, h, w)

    return final_core_mask, final_inflam_mask


# ==========================
# Synthetic day iterator
# ==========================
def compute_timeline(mode, pre_days, dev_days, static_days):
    """
    Returns: total_days, tuple_of_phases
    Phases are always in generated order.
    """
    if mode == "developing":
        return pre_days + dev_days, ("healthy", "developing")
    if mode == "static":
        return pre_days + static_days, ("healthy", "static")
    # "both" -> healthy -> developing -> static
    return pre_days + dev_days + static_days, ("healthy", "developing", "static")


# ==========================
# Per-variant runner
# ==========================
def run_variant_for_patient(mat_path, patient_id, variant_idx, base_params, mode, dev_days, static_days, out_root,
                            seed=None):
    rng_ = rng(seed)
    params = sample_variant_params(rng_)
    params.update(base_params)
    params["generation_mode"] = mode

    mat = scipy.io.loadmat(mat_path)
    left_crop = mat["Indirect_plantar_Right_crop"]  # project convention
    right_crop = mat["Indirect_plantar_Left_crop"]
    num_src_days = left_crop.shape[0]

    # ---- Reference corrected canvases + choose center/shape on Day 0 ----
    # OLD
    # ref_left, ref_right, info0 = correct_align_feet_for_day(left_crop[0, 0], right_crop[0, 0])
    # ref_h, ref_w = ref_left.shape

    # NEW
    ref_left, ref_right, tf = compute_patient_transform(left_crop, right_crop)
    ref_h, ref_w = ref_left.shape

    target_img = ref_left if params["apply_to"] == "left" else ref_right
    (y_center, x_center, shape_mode, _, _, region_full, region_key
     ) = select_center_and_shape_on_image(
        target_img, params["apply_wound_to"], params["position_mode"],
        params["manual_coord"], params["shape_mode"], rng_
    )

    # ---- Radii (relative) & coverage caps ----
    core_r, inflam_r = pick_final_radii(ref_h, ref_w, region_full, rng_, params)
    final_core_mask, final_inflam_mask, blob_count = build_final_mask(
        shape_mode, x_center, y_center,
        core_r, inflam_r,
        params["multi_min_blobs"], params["multi_max_blobs"],
        ref_h, ref_w
    )
    final_core_mask, final_inflam_mask = enforce_region_coverage_cap(
        final_core_mask, final_inflam_mask, region_full, ref_h, ref_w
    )

    # ---- Concise per-variant terminal update (non-spammy) ----
    if LOG_VARIANT_ASSIGNMENTS:
        extra = f", blobs={blob_count}" if shape_mode == "multi" else ""
        print(
            f"[patient={patient_id}] variant={variant_idx:02d} | "
            f"Dev={dev_days}, stat={static_days} | "
            f"target_foot={params['apply_to']}, region={region_key}, "
            f"center=({y_center},{x_center}), shape={shape_mode}{extra}"
        )

    # Output dirs
    total_days, phases = compute_timeline(mode, PRE_WOUND_DAYS, dev_days, static_days)
    subdir_root = os.path.join(
        out_root, f"{patient_id}", f"variant_{variant_idx:02d}",
        f"{mode}_pre{PRE_WOUND_DAYS}_dev{dev_days}_stat{static_days}"
    )
    subdir_png = os.path.join(subdir_root, "png")
    subdir_mat = os.path.join(subdir_root, "mat")
    os.makedirs(subdir_png, exist_ok=True)
    os.makedirs(subdir_mat, exist_ok=True)

    meta = {
        "patient": patient_id,
        "variant": int(variant_idx),
        "mode": mode,
        "dev_days": int(dev_days),
        "static_days": int(static_days),
        "apply_to": params["apply_to"],
        "apply_wound_to": params["apply_wound_to"],
        "region_key": region_key,
        "center": (int(y_center), int(x_center)),
        "shape_mode": shape_mode,
        "core_radius_final_px": int(core_r),
        "inflam_radius_final_px": int(inflam_r),
        "blur_sigma_core": float(params["blur_sigma_core"]),
        "blur_sigma_inflam": float(params["blur_sigma_inflam"]),
        "develop_mode": params["develop_mode"],
        "initial_size_scale": float(params["initial_size_scale"]),
        "initial_temp_scale": float(params["initial_temp_scale"]),
        "ref_canvas_shape": (int(ref_h), int(ref_w)),
        # OLD in meta:
        # "foot_correction_info_day0": info0,

        # NEW:
        "foot_correction_info_patient": {
            "rotation_angle": tf["angle"],
            "scale_x": tf["scale_x"],
            "scale_y": tf["scale_y"],
            "canvas_shape": tf["ref_shape"]
        },

        "size_policy": SIZE_POLICY,
        "coverage_caps": {
            "max_inflam_region_coverage": MAX_INFLAM_REGION_COVERAGE,
            "max_core_region_coverage": MAX_CORE_REGION_COVERAGE
        },
        "relative_ranges": {
            "core_radius_frac_range": CORE_RADIUS_FRAC_RANGE,
            "inflam_over_core_ratio": INFLAM_OVER_CORE_RATIO
        },
        "absolute_ranges_px": {
            "core_radius_px": ABS_CORE_RADIUS_RANGE,
            "inflam_radius_px": ABS_INFLAM_RADIUS_RANGE
        }
    }
    try:
        scipy.io.savemat(os.path.join(subdir_mat, "variant_metadata.mat"), {"metadata": meta})
    except Exception:
        pass

    # ---- Timeline ----
    for i in range(total_days):
        src_day = i % num_src_days

        # OLD
        # left_can, right_can, info_day = correct_align_feet_for_day(left_crop[src_day, 0], right_crop[src_day, 0])
        # if left_can.shape != (ref_h, ref_w):
        #     left_can = center_pad_to(left_can, ref_h, ref_w)
        #     right_can = center_pad_to(right_can, ref_h, ref_w)

        # NEW
        left_can, right_can, info_day = apply_patient_transform(
            left_crop[src_day, 0], right_crop[src_day, 0], tf
        )

        # Map global day index -> phase + progress within phase
        if mode == "both":
            if i < PRE_WOUND_DAYS:
                current_phase = "healthy"
                progress = 0.0
            elif i < PRE_WOUND_DAYS + dev_days:
                current_phase = "developing"
                progress = ((i - PRE_WOUND_DAYS + 1) / dev_days) if dev_days > 0 else 1.0
            else:
                current_phase = "static"
                progress = 1.0
        elif mode == "developing":
            if i < PRE_WOUND_DAYS:
                current_phase = "healthy"
                progress = 0.0
            else:
                current_phase = "developing"
                progress = ((i - PRE_WOUND_DAYS + 1) / dev_days) if dev_days > 0 else 1.0
        else:  # "static"
            if i < PRE_WOUND_DAYS:
                current_phase = "healthy"
                progress = 0.0
            else:
                current_phase = "static"
                progress = 1.0

        # ---- Build masks (or none) & apply wound only when needed ----
        if current_phase == "healthy":
            # no wound at all; keep canvases as-is
            core_mask = np.zeros((ref_h, ref_w), dtype=bool)
            inflam_mask = np.zeros((ref_h, ref_w), dtype=bool)
            increment = 0.0
            core_base = 0.0
            inflam_base = 0.0
            core_target = 0.0
            inflam_target = 0.0
            left_wounded, right_wounded = left_can, right_can
        else:
            core_mask, inflam_mask = masks_for_progress(
                progress, final_core_mask, final_inflam_mask, params, ref_h, ref_w
            )

            # choose which side gets the overlay (unchanged)
            if params["apply_to"] == "left":
                wounded_canvas = np.array(left_can, copy=True)
                normal_canvas = right_can
            else:
                wounded_canvas = np.array(right_can, copy=True)
                normal_canvas = left_can

            # contralateral-referenced base temps (unchanged)
            core_base = masked_nonzero_mean(normal_canvas, core_mask)
            inflam_base = masked_nonzero_mean(normal_canvas, inflam_mask)

            increment = FINAL_INCREMENT_DEG_C * (progress if current_phase == "developing" else 1.0)
            core_target = core_base + increment
            inflam_target = inflam_base + increment

            # soft-blend wound (unchanged)
            wounded_canvas = soft_blend_set(wounded_canvas, inflam_target, inflam_mask,
                                            sigma=params["blur_sigma_inflam"])
            wounded_canvas = soft_blend_set(wounded_canvas, core_target, core_mask, sigma=params["blur_sigma_core"])

            if params["apply_to"] == "left":
                left_wounded, right_wounded = wounded_canvas, right_can
            else:
                left_wounded, right_wounded = left_can, wounded_canvas

        # PNG debug (flip right back only for display)
        right_display = mirror_horiz(right_wounded)
        left_display = left_wounded
        gap = np.full((ref_h, GAP_COLS), np.nan)
        combined = np.hstack((left_display, gap, right_display))

        fig, ax = plt.subplots(figsize=FIGSIZE)
        im = ax.imshow(combined, cmap=CMAP)
        ax.axis("off")
        title = (
            f"{patient_id} • Var {variant_idx:02d} • Day {i + 1:02d} (src {src_day + 1:02d}) - {current_phase}\n"
            f"{params['apply_to']} / {region_key} | "
            f"core_base={core_base:.2f}°C→{core_target:.2f}°C, "
            f"inflam_base={inflam_base:.2f}°C→{inflam_target:.2f}°C, "
            f"Δ={increment:.2f}°C | core_r={core_r}px, inflam_r={inflam_r}px"
        )
        ax.set_title(title, fontsize=10)
        cbar = plt.colorbar(im, ax=ax, fraction=CBAR_FRACTION, pad=CBAR_PAD)
        cbar.set_label("Temperature (°C)")

        # --- Day-1 center marker (green dot) for quick visual debugging ---
        if MARK_CENTER_ON_FIRST_DAY and i == 0:
            # Compute where the center lands in the stitched display
            # Left display is not mirrored; right display is mirrored for aesthetics.
            if params["apply_to"] == "left":
                dot_x = x_center  # stays the same
                dot_y = y_center
                # no horizontal offset (it's on the left panel)
            else:
                # mirrored horizontally in the DISPLAY ONLY
                dot_x = (ref_w - 1 - x_center)  # mirror across width
                dot_y = y_center
                # shift by left panel width + gap to land in the right panel
                dot_x += ref_w + GAP_COLS

            # Draw the dot
            ax.scatter(
                [dot_x], [dot_y],
                s=CENTER_DOT_SIZE,
                c="lime",
                marker="o",
                edgecolors="black",
                linewidths=0.6,
                zorder=5,
            )

        out_png = os.path.join(
            subdir_png,
            f"{patient_id}_v{variant_idx:02d}_d{i + 1:02d}_src{src_day + 1:02d}_{current_phase}.png"
        )

        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close()

        # MAT save (one .mat per day with two cells: left & right)
        # Project convention:
        #   'Indirect_plantar_Right_crop'  -> LEFT foot (swapped naming)
        #   'Indirect_plantar_Left_crop'   -> RIGHT foot
        try:
            # Store the *wounded* canvases for that day (healthy days = same as original canvases)
            left_cell = np.empty((1, 1), dtype=object)
            right_cell = np.empty((1, 1), dtype=object)
            left_cell[0, 0] = left_wounded
            right_cell[0, 0] = right_wounded

            out_mat = os.path.join(
                subdir_mat,
                f"{patient_id}_v{variant_idx:02d}_day{i + 1:02d}.mat"
            )

            scipy.io.savemat(out_mat, {
                "left_crop": left_cell,  # LEFT foot (by project convention)
                "right_crop": right_cell,  # RIGHT foot
                # Keep a tiny bit of context if useful later:
                "phase": current_phase,
                "progress": float(progress),
            })
        except Exception as e:
            print(f"Warning: failed to save per-day mat for day {i + 1}: {e}")


# ==========================
# Entry
# ==========================
if __name__ == "__main__":
    if GLOBAL_SEED is not None:
        np.random.seed(GLOBAL_SEED)

    if RUN_ALL_PATIENTS:
        patient_ids = [f"{PATIENT_PREFIX}{i}" for i in range(1, PATIENT_COUNT + 1)]
    else:
        patient_ids = [SINGLE_PATIENT_ID]

    base_params = {
        "develop_mode": DEVELOP_MODE,
        "initial_size_scale": INITIAL_SIZE_SCALE,
        "initial_temp_scale": INITIAL_TEMP_SCALE,
    }

    for pid in patient_ids:
        mat_path = os.path.join(MAT_ROOT, f"{pid}.mat")
        for v in range(1, WOUND_VARIANTS_PER_PATIENT + 1):
            
            if GENERATION_MODE == "both":
                dev_days, static_days = generate_weighted_random(20, 10, 30), STATIC_DAYS
            elif GENERATION_MODE == "developing":
                dev_days, static_days = generate_weighted_random(20, 10, 30), 0
            elif GENERATION_MODE == "static":
                dev_days, static_days = 0, STATIC_DAYS
            else:
                raise ValueError("GENERATION_MODE must be 'static', 'developing', or 'both'")
            
            seed = None if GLOBAL_SEED is None else GLOBAL_SEED + (hash(pid) % 10_000) + v
            run_variant_for_patient(
                mat_path=mat_path,
                patient_id=pid,
                variant_idx=v,
                base_params=base_params,
                mode=GENERATION_MODE,
                dev_days=dev_days,
                static_days=static_days,
                out_root=OUTPUT_ROOT,
                seed=seed
            )