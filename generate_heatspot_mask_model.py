# --------------------------------------------
# Wound synthesis for single/all patients with:
# - Variable develop/static days (synthetic extension over limited frames)
# - Randomized wound target (left/right), region (heel/upper_foot), and position
# - Multiple wound variants per patient
# Keeps original logic/style, extends where needed.
# --------------------------------------------

import os
import warnings
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.io.matlab import MatReadWarning
from scipy.ndimage import gaussian_filter
from skimage.transform import rescale

# Segmentation utilities
from foot_part_identifier import segment_foot

warnings.filterwarnings("ignore", category=MatReadWarning)

# ==========================
# HIGH-LEVEL DATASET CONFIG
# ==========================
# Choose whether to run a single patient (gzX) or sweep gz1..gz15
RUN_ALL_PATIENTS = False
SINGLE_PATIENT_ID = "gz1"             # used only if RUN_ALL_PATIENTS=False
PATIENT_PREFIX = "gz"
PATIENT_COUNT  = 15                    # assume gz1..gz15 exist
MAT_ROOT = "Data/Temp Data"            # where gz*.mat live

# How many distinct wounds to generate per patient
WOUND_VARIANTS_PER_PATIENT = 3

# Output root
OUTPUT_ROOT = "output_images_wound_modes_segmented"

# Optional reproducibility
GLOBAL_SEED = None   # e.g., 123; set None to be fully random


# ==========================
# GENERATION/GROWTH CONFIG
# ==========================
# generation_mode: "static", "developing", or "both"
GENERATION_MODE = "both"

# For synthetic timelines (works for any number of available source days)
DEV_DAYS    = 20   # variable, default 20
STATIC_DAYS = 10   # variable, default 10
# If GENERATION_MODE is "developing", you get DEV_DAYS images.
# If "static", you get STATIC_DAYS images.
# If "both", you get DEV_DAYS + STATIC_DAYS images (first grow, then hold).

# Development behavior (unchanged defaults)
DEVELOP_MODE        = "size+intensity"  # "size+intensity" | "intensity-only"
INITIAL_SIZE_SCALE  = 0.05
INITIAL_TEMP_SCALE  = 0.05

# Visualization / formatting
GAP_COLS = 5
FIGSIZE = (8, 6)
CMAP = "hot"
CBAR_FRACTION = 0.035
CBAR_PAD = 0.04

# --------------------------------------------
# Randomization helpers (per-variant diversity)
# --------------------------------------------
def rng(seed=None):
    return np.random.default_rng(seed) if seed is not None else np.random.default_rng()

def sample_variant_params(rng_):
    """
    Randomize wound-side, region and placement-related aspects (plus a few extras).
    Kept conservative to preserve your original behavior.
    """
    params = {
        # required by request
        "apply_to": rng_.choice(["left", "right"]),
        "apply_wound_to": rng_.choice(["heel", "upper_foot"]),
        "position_mode": 2,                 # 2=random in target region
        "manual_coord": (150, 180),         # unused unless position_mode==3

        # wound geometry & thermals (kept near original defaults)
        "shape_mode": rng_.choice(["circle", "multi"]),
        "core_radius_final": int(rng_.integers(12, 21)),      # ~15 ±
        "inflam_radius_final": int(rng_.integers(24, 36)),    # ~30 ±
        "core_temp_final": float(rng_.uniform(3.5, 6.0)),     # ~5.0 ±
        "inflam_temp_final": float(rng_.uniform(2.0, 4.5)),   # ~3.0 ±
        "blur_sigma_core": float(rng_.uniform(5.0, 7.5)),     # ~6.0 ±
        "blur_sigma_inflam": float(rng_.uniform(5.0, 7.5)),   # ~6.0 ±
        "multi_min_blobs": 2,
        "multi_max_blobs": 6,

        # growth mode details
        "develop_mode": DEVELOP_MODE,
        "initial_size_scale": INITIAL_SIZE_SCALE,
        "initial_temp_scale": INITIAL_TEMP_SCALE,
    }
    return params


# ==========================
# Core helpers (kept close to original)
# ==========================
def to_display(img2d):
    """Map 0 -> NaN for visualization/overlay math."""
    return np.where(img2d == 0, np.nan, img2d)

def trim_empty_columns(img):
    """Remove fully-zero/NaN columns to bring feet closer."""
    valid_cols = ~(np.all(np.isnan(img) | (img == 0), axis=0))
    return img[:, valid_cols]

def build_final_mask(shape_mode, x_center, y_center, core_radius_final, inflam_radius_final,
                     multi_min_blobs, multi_max_blobs, h, w, rng_):
    """
    Final locked shape (core + inflammation). Re-implements your original,
    but without relying on global X/Y.
    """
    Y, X = np.ogrid[:h, :w]

    if shape_mode == "circle":
        final_core_mask   = (X - x_center)**2 + (Y - y_center)**2 <= core_radius_final**2
        final_inflam_mask = (X - x_center)**2 + (Y - y_center)**2 <= inflam_radius_final**2
        return final_core_mask, final_inflam_mask

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

    return core_mask_bin, inflam_mask_bin

def scale_mask(mask, scale_factor, h, w):
    """Scale a binary mask around its centroid (kept from your original idea)."""
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return mask
    cx, cy = np.mean(xs), np.mean(ys)

    # recenter to middle
    shift_x, shift_y = w//2 - cx, h//2 - cy
    shifted = np.roll(mask, (int(shift_y), int(shift_x)), axis=(0, 1)).astype(float)

    resized = rescale(shifted, scale=scale_factor, preserve_range=True,
                      anti_aliasing=True, order=1)

    rh, rw = resized.shape
    out = np.zeros((h, w), dtype=bool)
    sy = max((h - rh)//2, 0); sx = max((w - rw)//2, 0)
    ey = min(h, sy + rh);     ex = min(w, sx + rw)
    out[sy:ey, sx:ex] = resized[:ey-sy, :ex-sx] > 0.5

    # shift back
    final = np.roll(out, (-int(shift_y), -int(shift_x)), axis=(0, 1))
    return final

def make_mask(progress, final_core_mask, final_inflam_mask, params, h, w):
    """
    Build the per-day thermal overlay with blur (unchanged math, just parameterized).
    """
    generation_mode = params.get("generation_mode", "developing")
    develop_mode    = params["develop_mode"]

    core_val   = params["core_temp_final"]
    inflam_val = params["inflam_temp_final"]

    if generation_mode == "static":
        core_mask   = final_core_mask
        inflam_mask = final_inflam_mask
    else:
        if develop_mode == "size+intensity":
            scale_factor = params["initial_size_scale"] + (1.0 - params["initial_size_scale"]) * progress
            core_mask   = scale_mask(final_core_mask,   scale_factor, h, w)
            inflam_mask = scale_mask(final_inflam_mask, scale_factor, h, w)
            core_val   *= (params["initial_temp_scale"] + (1 - params["initial_temp_scale"]) * progress)
            inflam_val *= (params["initial_temp_scale"] + (1 - params["initial_temp_scale"]) * progress)
        elif develop_mode == "intensity-only":
            core_mask   = final_core_mask
            inflam_mask = final_inflam_mask
            core_val   *= (params["initial_temp_scale"] + (1 - params["initial_temp_scale"]) * progress)
            inflam_val *= (params["initial_temp_scale"] + (1 - params["initial_temp_scale"]) * progress)
        else:
            raise ValueError("Unknown develop_mode")

    mask_core   = np.zeros((h, w)); mask_core[core_mask]     = core_val
    mask_inflam = np.zeros((h, w)); mask_inflam[inflam_mask] = inflam_val

    mask_core   = gaussian_filter(mask_core,   sigma=params["blur_sigma_core"])
    mask_inflam = gaussian_filter(mask_inflam, sigma=params["blur_sigma_inflam"])
    return np.maximum(mask_inflam, mask_core)

def select_center_and_shape(left_crop, right_crop, apply_to, apply_wound_to,
                            position_mode, manual_coord, shape_mode, rng_):
    """
    Pick a wound center strictly inside the requested anatomical region
    using global image coordinates (no accidental heel hits).
    """
    # 1) Extract current day's images just to size and segment (day 0 is fine)
    scan_left  = left_crop[0, 0]
    scan_right = right_crop[0, 0]
    img_left   = np.where(scan_left == 0,  np.nan, scan_left)
    img_right  = np.where(scan_right == 0, np.nan, scan_right)

    # Which foot to target
    target_foot = img_left if apply_to == "left" else img_right
    h, w = target_foot.shape

    # 2) Segment target foot into slices
    heel_slice, mid_slice, upper_slice = segment_foot(target_foot)  # heel, mid, upper(bottom in array)
    # Sanity: widths should match
    if not (heel_slice.shape[1] == mid_slice.shape[1] == upper_slice.shape[1] == w):
        raise RuntimeError("segment_foot returned slices with mismatched widths.")

    # 3) Build a full-height mask for the requested region (global coords)
    region_key = _normalize_region_name(apply_wound_to)
    region_full = _build_full_region_mask(heel_slice, mid_slice, upper_slice, region_key)

    # 4) Sample strictly inside the region
    ys, xs = np.where(~np.isnan(region_full))
    if len(xs) == 0:
        # If the region is empty for some odd image, fall back to any non-NaN pixel of the target foot
        ys, xs = np.where(~np.isnan(target_foot))

    if position_mode == 1:
        y_center = int(np.mean(ys)); x_center = int(np.mean(xs))
    elif position_mode == 2:
        idx = int(rng_.integers(len(ys)))
        y_center, x_center = int(ys[idx]), int(xs[idx])
    elif position_mode == 3:
        y_center, x_center = manual_coord
    else:
        raise ValueError("Invalid position_mode")

    # Guard: ensure the chosen center sits in the intended region
    in_region = not np.isnan(region_full[y_center, x_center])
    if not in_region:
        # Try a few resamples before giving up
        for _ in range(25):
            idx = int(rng_.integers(len(ys)))
            y_try, x_try = int(ys[idx]), int(xs[idx])
            if not np.isnan(region_full[y_try, x_try]):
                y_center, x_center = y_try, x_try
                in_region = True
                break
    if not in_region:
        raise RuntimeError(f"Could not place center inside requested region '{region_key}'.")

    # Shape choice (keep your variety)
    chosen_shape = shape_mode if shape_mode in ["circle", "multi"] else rng_.choice(["circle", "multi"])

    # Debug log (helps when auditing results)
    print(f"[select_center_and_shape] target_foot={apply_to}, region={region_key}, center=({y_center},{x_center}), shape={chosen_shape}")

    return y_center, x_center, chosen_shape, h, w

def _normalize_region_name(name: str) -> str:
    """Map user-friendly aliases to canonical region keys."""
    n = (name or "").strip().lower()
    if n in {"heel", "calcaneus"}:
        return "heel"
    if n in {"upper", "upper_foot", "upperfoot", "toes", "forefoot"}:
        return "upper_foot"
    if n in {"mid", "mid_foot", "midfoot", "arch"}:
        return "mid_foot"
    # default to upper_foot (safer for training variety)
    return "upper_foot"

def _build_full_region_mask(heel, mid, upper, region_key):
    """Recreate a full-height mask where only the target slice is kept."""
    # All three share the same width; heights sum to original height.
    nan_heel = np.full_like(heel, np.nan)
    nan_mid  = np.full_like(mid,  np.nan)
    nan_up   = np.full_like(upper, np.nan)

    if region_key == "heel":
        full = np.vstack((heel, nan_mid, nan_up))
    elif region_key == "mid_foot":
        full = np.vstack((nan_heel, mid, nan_up))
    elif region_key == "upper_foot":
        full = np.vstack((nan_heel, nan_mid, upper))
    else:
        # Should never happen, but avoid surprises
        full = np.vstack((nan_heel, nan_mid, upper))
    return full

# ==========================
# Synthetic day iterator
# ==========================
def compute_timeline(mode, dev_days, static_days):
    if mode == "developing":
        return dev_days, ("developing",)
    if mode == "static":
        return static_days, ("static",)
    # "both"
    return dev_days + static_days, ("developing", "static")


# ==========================
# Per-variant runner
# ==========================
def run_variant_for_patient(mat_path, patient_id, variant_idx, base_params, mode, dev_days, static_days, out_root, seed=None):
    rng_ = rng(seed)

    # Randomize the variant
    params = sample_variant_params(rng_)
    params.update(base_params)  # allow global develop_mode / scales
    params["generation_mode"] = mode

    # Load data
    mat = scipy.io.loadmat(mat_path)
    left_crop  = mat["Indirect_plantar_Right_crop"]  # (project convention)
    right_crop = mat["Indirect_plantar_Left_crop"]

    num_src_days = left_crop.shape[0]  # typically 10

    # Choose center + shape
    y_center, x_center, shape_mode, h, w = select_center_and_shape(
        left_crop, right_crop,
        params["apply_to"], params["apply_wound_to"],
        params["position_mode"], params["manual_coord"],
        params["shape_mode"], rng_
    )

    # Build final (locked) shape
    final_core_mask, final_inflam_mask = build_final_mask(
        shape_mode, x_center, y_center,
        params["core_radius_final"], params["inflam_radius_final"],
        params["multi_min_blobs"], params["multi_max_blobs"],
        h, w, rng_
    )

    # Prepare output dir
    total_days, phases = compute_timeline(mode, dev_days, static_days)
    subdir = os.path.join(
        out_root,
        f"{patient_id}",
        f"variant_{variant_idx:02d}",
        f"{mode}_dev{dev_days}_stat{static_days}"
    )
    os.makedirs(subdir, exist_ok=True)

    # Save a tiny metadata file for traceability
    meta = {
        "patient": patient_id,
        "variant": int(variant_idx),
        "mode": mode,
        "dev_days": int(dev_days),
        "static_days": int(static_days),
        "apply_to": params["apply_to"],
        "apply_wound_to": params["apply_wound_to"],
        "position_mode": params["position_mode"],
        "center": (int(y_center), int(x_center)),
        "shape_mode": shape_mode,
        "core_radius_final": int(params["core_radius_final"]),
        "inflam_radius_final": int(params["inflam_radius_final"]),
        "core_temp_final": float(params["core_temp_final"]),
        "inflam_temp_final": float(params["inflam_temp_final"]),
        "blur_sigma_core": float(params["blur_sigma_core"]),
        "blur_sigma_inflam": float(params["blur_sigma_inflam"]),
        "develop_mode": params["develop_mode"],
        "initial_size_scale": float(params["initial_size_scale"]),
        "initial_temp_scale": float(params["initial_temp_scale"]),
    }
    try:
        scipy.io.savemat(os.path.join(subdir, "variant_metadata.mat"), {"metadata": meta})
    except Exception:
        pass

    # Main timeline loop with synthetic day extension
    for i in range(total_days):
        src_day = i % num_src_days  # 0..9 cycling, extends timeline synthetically

        if mode == "both":
            current_phase = "developing" if i < dev_days else "static"
            progress = (i + 1) / dev_days if i < dev_days and dev_days > 0 else 1.0
        elif mode == "developing":
            current_phase = "developing"
            progress = (i + 1) / dev_days if dev_days > 0 else 1.0
        else:  # "static"
            current_phase = "static"
            progress = 1.0

        # Make overlay for this (synthetic) day
        mask_day = make_mask(progress, final_core_mask, final_inflam_mask, params, h, w)

        scan_left  = left_crop[src_day, 0]
        scan_right = right_crop[src_day, 0]
        img_left   = to_display(scan_left)
        img_right  = to_display(scan_right)

        if params["apply_to"] == "left":
            img_left_w  = np.where(~np.isnan(img_left),  img_left  + mask_day, np.nan)
            img_right_w = img_right
        else:
            img_left_w  = img_left
            img_right_w = np.where(~np.isnan(img_right), img_right + mask_day, np.nan)

        img_left_trim  = trim_empty_columns(img_left_w)
        img_right_trim = trim_empty_columns(img_right_w)

        gap = np.full((img_left_trim.shape[0], GAP_COLS), np.nan)
        combined = np.hstack((img_left_trim, gap, img_right_trim))

        # Figure
        fig, ax = plt.subplots(figsize=FIGSIZE)
        im = ax.imshow(combined, cmap=CMAP)
        ax.axis("off")
        title = (f"{patient_id} • Var {variant_idx:02d} • Day {i+1:02d} "
                 f"(src {src_day+1:02d}) — {current_phase} ({params['apply_to']}, {params['apply_wound_to']})")
        ax.set_title(title)
        cbar = fig.colorbar(im, ax=ax, fraction=CBAR_FRACTION, pad=CBAR_PAD)
        cbar.set_label("Temperature (°C)")

        out_png = os.path.join(subdir, f"{patient_id}_v{variant_idx:02d}_d{i+1:02d}_src{src_day+1:02d}_{current_phase}.png")
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close()

        # Save MAT for training pipelines (trimmed feet + wound mask)
        try:
            scipy.io.savemat(
                os.path.join(subdir, f"{patient_id}_v{variant_idx:02d}_d{i+1:02d}_{current_phase}.mat"),
                {
                    "left_foot":  img_left_trim,
                    "right_foot": img_right_trim,
                    "wound_mask": mask_day,
                    "phase": current_phase,
                    "progress": float(progress),
                }
            )
        except Exception:
            pass


# ==========================
# Entry point
# ==========================
if __name__ == "__main__":
    if GLOBAL_SEED is not None:
        np.random.seed(GLOBAL_SEED)

    # Determine patients
    if RUN_ALL_PATIENTS:
        patient_ids = [f"{PATIENT_PREFIX}{i}" for i in range(1, PATIENT_COUNT + 1)]
    else:
        patient_ids = [SINGLE_PATIENT_ID]

    # How many days to output
    if GENERATION_MODE == "both":
        dev_days    = int(DEV_DAYS)
        static_days = int(STATIC_DAYS)
    elif GENERATION_MODE == "developing":
        dev_days, static_days = int(DEV_DAYS), 0
    elif GENERATION_MODE == "static":
        dev_days, static_days = 0, int(STATIC_DAYS)
    else:
        raise ValueError("GENERATION_MODE must be 'static', 'developing', or 'both'")

    # Base (non-random) settings inherited by every variant
    base_params = {
        "develop_mode": DEVELOP_MODE,
        "initial_size_scale": INITIAL_SIZE_SCALE,
        "initial_temp_scale": INITIAL_TEMP_SCALE,
    }

    # Run
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
