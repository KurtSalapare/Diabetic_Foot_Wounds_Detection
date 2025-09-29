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
import scipy.io
import matplotlib.pyplot as plt
from scipy.io.matlab import MatReadWarning
from scipy.ndimage import gaussian_filter
from skimage.transform import rescale

from foot_part_identifier import segment_foot
from foot_overlay_creator import (
    to_nan, trim_to_content, mirror_horiz, pad_to_same_size,
    rotate_image_preserve_temps, find_best_rotation_angle,
    scale_image_preserve_temps
)

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
SINGLE_PATIENT_ID = "gz1"
PATIENT_PREFIX = "gz"
PATIENT_COUNT  = 15
MAT_ROOT = "Data/Temp Data"

WOUND_VARIANTS_PER_PATIENT = 3
OUTPUT_ROOT = "output_images_wound_modes_segmented"

GLOBAL_SEED = None  # e.g., 123

# Log one concise line per variant
LOG_VARIANT_ASSIGNMENTS = True

# ==========================
# GENERATION/GROWTH CONFIG
# ==========================
GENERATION_MODE = "both"   # "static", "developing", "both"
DEV_DAYS    = 20
STATIC_DAYS = 10

DEVELOP_MODE        = "size+intensity"  # "size+intensity" | "intensity-only"
INITIAL_SIZE_SCALE  = 0.05
INITIAL_TEMP_SCALE  = 0.05  # kept for compatibility (not used by new temp evolution)

# ---- Relative size policy (prevents huge wounds on padded canvases) ----
SIZE_POLICY = "relative"    # "relative" (recommended) | "absolute"

# Relative sizing (fractions of min(ref_h, ref_w))
CORE_RADIUS_FRAC_RANGE = (0.025, 0.040)     # ~2.5%..4.0% of min dimension
INFLAM_OVER_CORE_RATIO = (1.6, 2.2)         # inflam radius = core * ratio
# Max final coverage (fraction of target region)
MAX_INFLAM_REGION_COVERAGE = 0.08           # 8%
MAX_CORE_REGION_COVERAGE   = 0.03           # 3%

# Absolute sizing fallback (narrowed)
ABS_CORE_RADIUS_RANGE   = (8, 15)
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
        "position_mode": 2,                 # random in region
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
def to_display(img2d):
    return np.where(img2d == 0, np.nan, img2d)

def _build_full_region_mask(heel, mid, upper, region_key):
    """Return a full-height mask for exactly one of: 'heel', 'mid_foot', 'upper_foot'."""
    nan_heel = np.full_like(heel, np.nan)
    nan_mid  = np.full_like(mid,  np.nan)
    nan_up   = np.full_like(upper, np.nan)

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
    """Pick a wound center strictly inside the requested anatomical region."""
    if apply_wound_to not in {"heel", "mid_foot", "upper_foot"}:
        raise ValueError(
            f"apply_wound_to must be one of 'heel', 'mid_foot', 'upper_foot' (got: {apply_wound_to!r})"
        )

    heel, mid_foot, upper_foot = segment_foot(target_foot_img)
    h, w = target_foot_img.shape

    region_key = apply_wound_to  # use exactly what caller provided
    region_full = _build_full_region_mask(heel, mid_foot, upper_foot, region_key)

    ys, xs = np.where(~np.isnan(region_full))
    if len(xs) == 0:
        # Fall back to any valid pixel on the target foot image (rare)
        ys, xs = np.where(~np.isnan(target_foot_img))

    if position_mode == 1:
        y_center = int(np.mean(ys)); x_center = int(np.mean(xs))
    elif position_mode == 2:
        idx = int(rng_.integers(len(ys)))
        y_center, x_center = int(ys[idx]), int(xs[idx])
    elif position_mode == 3:
        y_center, x_center = manual_coord
    else:
        raise ValueError("Invalid position_mode")

    chosen_shape = shape_mode if shape_mode in ["circle", "multi"] else rng_.choice(["circle", "multi"])
    return y_center, x_center, chosen_shape, h, w, region_full, region_key

def build_final_mask(shape_mode, x_center, y_center, core_radius_final, inflam_radius_final,
                     multi_min_blobs, multi_max_blobs, h, w, rng_):
    """
    Returns: final_core_mask, final_inflam_mask, blob_count
    - blob_count = 1 for 'circle', or the sampled number of blobs for 'multi'
    """
    Y, X = np.ogrid[:h, :w]

    if shape_mode == "circle":
        final_core_mask   = (X - x_center)**2 + (Y - y_center)**2 <= core_radius_final**2
        final_inflam_mask = (X - x_center)**2 + (Y - y_center)**2 <= inflam_radius_final**2
        return final_core_mask, final_inflam_mask, 1

    # multi-blob union
    n_blobs = int(rng_.integers(multi_min_blobs, multi_max_blobs + 1))
    core_mask_bin = np.zeros((h, w), dtype=bool)
    inflam_mask_bin = np.zeros((h, w), dtype=bool)

    centers = [(x_center, y_center)]
    core_r_list = [core_radius_final]
    inflam_r_list = [inflam_radius_final]

    for _ in range(n_blobs - 1):
        base_x, base_y = centers[int(rng_.integers(len(centers)))]
        angle = rng_.uniform(0, 2*np.pi)
        dist  = int(rng_.integers(max(2, core_radius_final//2), max(3, core_radius_final*2)))
        dx, dy = int(np.cos(angle) * dist), int(np.sin(angle) * dist)
        new_x = int(np.clip(base_x + dx, 0, w-1))
        new_y = int(np.clip(base_y + dy, 0, h-1))
        centers.append((new_x, new_y))
        core_r_list.append(int(rng_.integers(max(2, core_radius_final//2), max(3, core_radius_final))))
        inflam_r_list.append(int(rng_.integers(max(2, inflam_radius_final//2), max(3, inflam_radius_final))))

    for (cx, cy), rc, ri in zip(centers, core_r_list, inflam_r_list):
        core_mask_bin   |= (X - cx)**2 + (Y - cy)**2 <= rc**2
        inflam_mask_bin |= (X - cx)**2 + (Y - cy)**2 <= ri**2

    return core_mask_bin, inflam_mask_bin, n_blobs

def scale_mask(mask, scale_factor, h, w):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return mask
    cx, cy = np.mean(xs), np.mean(ys)
    shift_x, shift_y = w//2 - cx, h//2 - cy
    shifted = np.roll(mask, (int(shift_y), int(shift_x)), axis=(0, 1)).astype(float)
    resized = rescale(shifted, scale=scale_factor, preserve_range=True, anti_aliasing=True, order=1)
    rh, rw = resized.shape
    out = np.zeros((h, w), dtype=bool)
    sy = max((h - rh)//2, 0); sx = max((w - rw)//2, 0)
    ey = min(h, sy + rh);     ex = min(w, sx + rw)
    out[sy:ey, sx:ex] = resized[:ey-sy, :ex-sx] > 0.5
    final = np.roll(out, (-int(shift_y), -int(shift_x)), axis=(0, 1))
    return final

def masks_for_progress(progress, final_core_mask, final_inflam_mask, params, h, w):
    if params["develop_mode"] == "size+intensity":
        scale_factor = params["initial_size_scale"] + (1.0 - params["initial_size_scale"]) * progress
        core_mask   = scale_mask(final_core_mask,   scale_factor, h, w)
        inflam_mask = scale_mask(final_inflam_mask, scale_factor, h, w)
    elif params["develop_mode"] == "intensity-only":
        core_mask   = final_core_mask
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

# ---------------------------
# Feet correction per day
# ---------------------------
def correct_align_feet_for_day(scan_left, scan_right):
    with _silence_imported(SUPPRESS_IMPORTED_LOGS):
        img_left  = trim_to_content(to_nan(scan_left,  adaptive_threshold=True))
        img_right = trim_to_content(to_nan(scan_right, adaptive_threshold=True))
        img_right_mir = mirror_horiz(img_right)
        angle, score, _scores = find_best_rotation_angle(img_left, img_right_mir)
        if angle != 0:
            img_right_mir = rotate_image_preserve_temps(img_right_mir, angle)
        if img_right_mir.shape != img_left.shape:
            scale_x = img_left.shape[1] / img_right_mir.shape[1]
            scale_y = img_left.shape[0] / img_right_mir.shape[0]
            img_right_scaled = scale_image_preserve_temps(img_right_mir, scale_x, scale_y)
        else:
            img_right_scaled = img_right_mir
            scale_x = scale_y = 1.0
        left_canvas, right_canvas = pad_to_same_size(img_left, img_right_scaled)

    info = {"rotation_angle": float(angle), "overlap_score": float(score),
            "scale_x": float(scale_x), "scale_y": float(scale_y),
            "canvas_shape": left_canvas.shape}
    return left_canvas, right_canvas, info

def center_pad_to(img, target_h, target_w):
    h, w = img.shape
    canvas = np.full((target_h, target_w), np.nan, dtype=float)
    top = max(0, (target_h - h) // 2)
    left = max(0, (target_w - w) // 2)
    canvas[top:top+h, left:left+w] = img
    return canvas

# ---- Radii selection & caps ----
def pick_final_radii(h, w, region_full, rng_, params):
    if SIZE_POLICY == "relative":
        min_dim = float(min(h, w))
        core_frac = rng_.uniform(*CORE_RADIUS_FRAC_RANGE)
        core_r = int(max(2, round(min_dim * core_frac)))
        ratio = rng_.uniform(*INFLAM_OVER_CORE_RATIO)
        inflam_r = int(max(core_r + 1, round(core_r * ratio)))
    else:
        core_r   = int(rng_.integers(*ABS_CORE_RADIUS_RANGE))
        inflam_r = int(rng_.integers(*ABS_INFLAM_RADIUS_RANGE))
        inflam_r = max(inflam_r, core_r + 1)
    return core_r, inflam_r

def enforce_region_coverage_cap(final_core_mask, final_inflam_mask, region_full, h, w):
    region_area = np.count_nonzero(~np.isnan(region_full))
    if region_area == 0:
        return final_core_mask, final_inflam_mask
    inflam_area = int(np.count_nonzero(final_inflam_mask))
    core_area   = int(np.count_nonzero(final_core_mask))
    inflam_frac = inflam_area / region_area
    core_frac   = core_area / region_area

    scale_needed = 1.0
    if inflam_frac > MAX_INFLAM_REGION_COVERAGE:
        scale_needed = min(scale_needed, np.sqrt(MAX_INFLAM_REGION_COVERAGE / max(inflam_frac, 1e-8)))
    if core_frac > MAX_CORE_REGION_COVERAGE:
        scale_needed = min(scale_needed, np.sqrt(MAX_CORE_REGION_COVERAGE / max(core_frac, 1e-8)))

    if scale_needed < 1.0:
        final_inflam_mask = scale_mask(final_inflam_mask, scale_needed, h, w)
        final_core_mask   = scale_mask(final_core_mask,   scale_needed, h, w)

    return final_core_mask, final_inflam_mask

# ==========================
# Synthetic day iterator
# ==========================
def compute_timeline(mode, dev_days, static_days):
    if mode == "developing":
        return dev_days, ("developing",)
    if mode == "static":
        return static_days, ("static",)
    return dev_days + static_days, ("developing", "static")

# ==========================
# Per-variant runner
# ==========================
def run_variant_for_patient(mat_path, patient_id, variant_idx, base_params, mode, dev_days, static_days, out_root, seed=None):
    rng_ = rng(seed)
    params = sample_variant_params(rng_)
    params.update(base_params)
    params["generation_mode"] = mode

    mat = scipy.io.loadmat(mat_path)
    left_crop  = mat["Indirect_plantar_Right_crop"]   # project convention
    right_crop = mat["Indirect_plantar_Left_crop"]
    num_src_days = left_crop.shape[0]

    # ---- Reference corrected canvases + choose center/shape on Day 0 ----
    ref_left, ref_right, info0 = correct_align_feet_for_day(left_crop[0,0], right_crop[0,0])
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
        ref_h, ref_w, rng_
    )
    final_core_mask, final_inflam_mask = enforce_region_coverage_cap(
        final_core_mask, final_inflam_mask, region_full, ref_h, ref_w
    )

    # ---- Concise per-variant terminal update (non-spammy) ----
    if LOG_VARIANT_ASSIGNMENTS:
        extra = f", blobs={blob_count}" if shape_mode == "multi" else ""
        print(
            f"[patient={patient_id}] variant={variant_idx:02d} | "
            f"target_foot={params['apply_to']}, region={region_key}, "
            f"center=({y_center},{x_center}), shape={shape_mode}{extra}"
        )

    # Output dirs
    total_days, phases = compute_timeline(mode, dev_days, static_days)
    subdir_root = os.path.join(
        out_root, f"{patient_id}", f"variant_{variant_idx:02d}",
        f"{mode}_dev{dev_days}_stat{static_days}"
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
        "foot_correction_info_day0": info0,
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

        left_can, right_can, info_day = correct_align_feet_for_day(left_crop[src_day,0], right_crop[src_day,0])
        if left_can.shape != (ref_h, ref_w):
            left_can  = center_pad_to(left_can,  ref_h, ref_w)
            right_can = center_pad_to(right_can, ref_h, ref_w)

        if mode == "both":
            current_phase = "developing" if i < dev_days else "static"
            progress = ((i + 1) / dev_days) if i < dev_days and dev_days > 0 else 1.0
        elif mode == "developing":
            current_phase = "developing"
            progress = ((i + 1) / dev_days) if dev_days > 0 else 1.0
        else:
            current_phase = "static"
            progress = 1.0

        core_mask, inflam_mask = masks_for_progress(progress, final_core_mask, final_inflam_mask, params, ref_h, ref_w)

        if params["apply_to"] == "left":
            wounded_canvas = np.array(left_can,  copy=True)
            normal_canvas  = right_can
        else:
            wounded_canvas = np.array(right_can, copy=True)
            normal_canvas  = left_can

        core_base = masked_nonzero_mean(normal_canvas, core_mask)
        inflam_base = masked_nonzero_mean(normal_canvas, inflam_mask)

        increment = FINAL_INCREMENT_DEG_C * (progress if current_phase == "developing" else 1.0)
        core_target   = core_base   + increment
        inflam_target = inflam_base + increment

        wounded_canvas = soft_blend_set(wounded_canvas, inflam_target, inflam_mask, sigma=params["blur_sigma_inflam"])
        wounded_canvas = soft_blend_set(wounded_canvas, core_target,   core_mask,   sigma=params["blur_sigma_core"])

        if params["apply_to"] == "left":
            left_wounded, right_wounded = wounded_canvas, right_can
        else:
            left_wounded, right_wounded = left_can, wounded_canvas

        # PNG debug (flip right back only for display)
        right_display = mirror_horiz(right_wounded)
        left_display  = left_wounded
        gap = np.full((ref_h, GAP_COLS), np.nan)
        combined = np.hstack((left_display, gap, right_display))

        fig, ax = plt.subplots(figsize=FIGSIZE)
        im = ax.imshow(combined, cmap=CMAP)
        ax.axis("off")
        title = (
            f"{patient_id} • Var {variant_idx:02d} • Day {i+1:02d} (src {src_day+1:02d}) — {current_phase}\n"
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

        # MAT save
        try:
            scipy.io.savemat(
                os.path.join(
                    subdir_mat,
                    f"{patient_id}_v{variant_idx:02d}_d{i + 1:02d}_{current_phase}.mat"
                ),
                {
                    "left_canvas": left_can,
                    "right_canvas": right_can,
                    "left_wounded": left_wounded,
                    "right_wounded": right_wounded,
                    "core_mask": core_mask,
                    "inflam_mask": inflam_mask,
                    "core_base": float(core_base),
                    "inflam_base": float(inflam_base),
                    "increment": float(increment),
                    "core_target": float(core_target),
                    "inflam_target": float(inflam_target),
                    "phase": current_phase,
                    "progress": float(progress),
                    "foot_correction_info": info_day
                }
            )
        except Exception:
            pass

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

    if GENERATION_MODE == "both":
        dev_days, static_days = int(DEV_DAYS), int(STATIC_DAYS)
    elif GENERATION_MODE == "developing":
        dev_days, static_days = int(DEV_DAYS), 0
    elif GENERATION_MODE == "static":
        dev_days, static_days = 0, int(STATIC_DAYS)
    else:
        raise ValueError("GENERATION_MODE must be 'static', 'developing', or 'both'")

    base_params = {
        "develop_mode": DEVELOP_MODE,
        "initial_size_scale": INITIAL_SIZE_SCALE,
        "initial_temp_scale": INITIAL_TEMP_SCALE,
    }

    for pid in patient_ids:
        mat_path = os.path.join(MAT_ROOT, f"{pid}.mat")
        for v in range(1, WOUND_VARIANTS_PER_PATIENT + 1):
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
