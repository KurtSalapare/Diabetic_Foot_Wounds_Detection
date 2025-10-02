import os
import glob
import warnings
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.io.matlab.miobase import MatReadWarning

# ==========================
# PIPELINE INTEGRATION (NEW)
# ==========================
# INPUT: wound-generation outputs (unchanged from the previous step)
INPUT_ROOT = "output_images_wound_modes_segmented"

# OUTPUT: write detection PNGs into a new top-level folder that mirrors the structure
#         <DETECT_OUTPUT_ROOT>/<patient>/<variant>/<run>/png/wound_detect_dayN.png
DETECT_OUTPUT_ROOT = "output_images_detect_wounds"

PATIENT_FILTER = None      # e.g., "gz7" or ["gz7","gz8"]; None = all discovered
VARIANT_FILTER = None      # e.g., "variant_01" or ["variant_01"]; None = all
RUNDIR_FILTER  = None      # e.g., "both_pre10_dev20_stat10"; None = all

# Which day indices (0-based). None = all days found.
days_to_process = None

# ==========================
# DETECTION CONFIG (unchanged)
# ==========================
chunk_px = 5
gap_cols = 5
temp_diff_threshold = 2.2
figsize = (8, 6)
cbar_fraction = 0.035
cbar_pad = 0.04
colormap = "hot"

warnings.filterwarnings("ignore", category=MatReadWarning)

# ---------- Helpers (logic UNCHANGED) ----------

def trim_empty_columns(img):
    valid_cols = ~(np.all(np.isnan(img) | (img == 0), axis=0))
    return img[:, valid_cols], valid_cols

def make_display_image(scan_raw):
    return np.where(scan_raw == 0, np.nan, scan_raw)

def compute_valid_grid_boxes(scan_raw, img_trimmed, valid_cols_bool, chunk_px):
    scan_raw_trimmed = scan_raw[:, valid_cols_bool]
    h, w = img_trimmed.shape
    boxes = []
    indices = []
    for y0 in range(0, h - chunk_px + 1, chunk_px):
        for x0 in range(0, w - chunk_px + 1, chunk_px):
            y1, x1 = y0 + chunk_px, x0 + chunk_px
            region_trim = img_trimmed[y0:y1, x0:x1]
            if np.isnan(region_trim).any():
                continue
            region_raw_trim = scan_raw_trimmed[y0:y1, x0:x1]
            if np.any(region_raw_trim == 0):
                continue
            boxes.append((x0, y0, chunk_px, chunk_px))
            indices.append((x0 // chunk_px, y0 // chunk_px))
    return boxes, indices

def foot_center_chunk(img_trimmed, boxes, indices, chunk_px):
    ys, xs = np.where(~np.isnan(img_trimmed))
    if len(xs) == 0:
        return None, None
    cx = np.mean(xs); cy = np.mean(ys)
    icx = int(cx // chunk_px); icy = int(cy // chunk_px)
    valid_set = set(indices)
    if (icx, icy) in valid_set:
        for (x0, y0, w, h), (ix, iy) in zip(boxes, indices):
            if ix == icx and iy == icy:
                return (icx, icy), (x0, y0)
    if len(indices) == 0:
        return None, None
    idx_arr = np.array(indices)
    d2 = (idx_arr[:,0] - icx)**2 + (idx_arr[:,1] - icy)**2
    k = int(np.argmin(d2))
    icx2, icy2 = indices[k]
    x0, y0, _, _ = boxes[k]
    return (icx2, icy2), (x0, y0)

def compute_extents(indices, center_idx):
    if center_idx is None or len(indices) == 0:
        return 1, 1
    icx, icy = center_idx
    dx = [ix - icx for ix, iy in indices]
    dy = [iy - icy for ix, iy in indices]
    max_abs_dx = max(1, int(np.max(np.abs(dx))))
    max_abs_dy = max(1, int(np.max(np.abs(dy))))
    return max_abs_dx, max_abs_dy

def nearest_valid_index(target_ix, target_iy, indices):
    if len(indices) == 0:
        return None
    idx_arr = np.array(indices)
    d2 = (idx_arr[:,0] - target_ix)**2 + (idx_arr[:,1] - target_iy)**2
    k = int(np.argmin(d2))
    return tuple(idx_arr[k])

def map_chunk_across_feet(src_idx, src_center_idx, dst_center_idx,
                          src_extents, dst_extents, mirror=True):
    if src_center_idx is None or dst_center_idx is None:
        return None
    (src_cx, src_cy) = src_center_idx
    (dst_cx, dst_cy) = dst_center_idx
    (src_madx, src_mady) = src_extents
    (dst_madx, dst_mady) = dst_extents
    ix, iy = src_idx
    dx = ix - src_cx
    dy = iy - src_cy
    sx = dst_madx / max(src_madx, 1)
    sy = dst_mady / max(src_mady, 1)
    mdx = -dx * sx if mirror else dx * sx
    mdy =  dy * sy
    mapped_ix = int(round(dst_cx + mdx))
    mapped_iy = int(round(dst_cy + mdy))
    return (mapped_ix, mapped_iy)

def build_index_to_rect(indices, boxes):
    d = {}
    for (x0, y0, w0, h0), (ix0, iy0) in zip(boxes, indices):
        d[(ix0, iy0)] = (x0, y0, w0, h0)
    return d

# ---------- NEW: I/O helpers (non-logic) ----------

def _discover_runs(root):
    if not os.path.isdir(root):
        return
    patients = sorted(os.listdir(root))
    if PATIENT_FILTER is not None:
        want = set([PATIENT_FILTER] if isinstance(PATIENT_FILTER, str) else PATIENT_FILTER)
        patients = [p for p in patients if p in want]
    for pid in patients:
        pdir = os.path.join(root, pid)
        if not os.path.isdir(pdir):
            continue
        variants = sorted([d for d in os.listdir(pdir) if d.startswith("variant_")])
        if VARIANT_FILTER is not None:
            wantv = set([VARIANT_FILTER] if isinstance(VARIANT_FILTER, str) else VARIANT_FILTER)
            variants = [v for v in variants if v in wantv]
        for v in variants:
            vdir = os.path.join(pdir, v)
            runs = sorted([d for d in os.listdir(vdir) if os.path.isdir(os.path.join(vdir, d))])
            if RUNDIR_FILTER is not None:
                wantr = set([RUNDIR_FILTER] if isinstance(RUNDIR_FILTER, str) else RUNDIR_FILTER)
                runs = [r for r in runs if r in wantr]
            for r in runs:
                yield pid, v, os.path.join(vdir, r)

def _load_day_mat(path):
    m = scipy.io.loadmat(path)
    phase = m.get("phase", None)
    progress = m.get("progress", None)
    if "left_crop" in m and "right_crop" in m:
        left = m["left_crop"][0, 0]
        right = m["right_crop"][0, 0]
    elif "left_foot" in m and "right_foot" in m:
        left = m["left_foot"]
        right = m["right_foot"]
    else:
        raise KeyError(f"No left/right in {os.path.basename(path)}")
    # Flip right back to anatomical orientation (generator saved corrected/mirrored canvas)
    right = np.fliplr(right)
    return left, right, phase, progress

# ---------- MAIN ----------

def main():
    any_found = False
    for patient_id, variant_dir, run_path in _discover_runs(INPUT_ROOT):
        mat_dir = os.path.join(run_path, "mat")
        if not os.path.isdir(mat_dir):
            continue

        day_files = sorted(glob.glob(os.path.join(mat_dir, f"{patient_id}_v*_day*.mat")))
        if not day_files:
            continue

        # Select days
        use_idx = range(len(day_files)) if days_to_process is None else [
            i for i in days_to_process if 0 <= i < len(day_files)
        ]

        # NEW: mirror the hierarchy into a separate output root
        run_name = os.path.basename(run_path)
        out_dir = os.path.join(DETECT_OUTPUT_ROOT, patient_id, variant_dir, run_name, "png")
        os.makedirs(out_dir, exist_ok=True)

        # Single concise variant/run print (no per-day prints)
        print(f"[detect] patient={patient_id} | {variant_dir} | run='{run_name}' | days={len(list(use_idx))} -> out='{os.path.relpath(out_dir)}'")

        # Process each selected day with UNCHANGED detection logic
        for di in use_idx:
            day_mat = day_files[di]

            # --- PER-DAY PRINT (START) ---
            print(f"   → day {di + 1:02d}: {os.path.basename(day_mat)}")

            try:
                scan_left_raw, scan_right_raw, phase, progress = _load_day_mat(day_mat)
            except Exception as e:
                print(f"   ! skip {os.path.basename(day_mat)}: {e}")
                continue

            # === BEGIN: detection logic (unchanged) ===
            img_left  = make_display_image(scan_left_raw)
            img_right = make_display_image(scan_right_raw)

            img_left_trim,  valid_cols_left  = trim_empty_columns(img_left)
            img_right_trim, valid_cols_right = trim_empty_columns(img_right)

            boxes_left,  idx_left  = compute_valid_grid_boxes(scan_left_raw,  img_left_trim,  valid_cols_left,  chunk_px)
            boxes_right, idx_right = compute_valid_grid_boxes(scan_right_raw, img_right_trim, valid_cols_right, chunk_px)

            (center_idx_left,  center_xy_left)  = foot_center_chunk(img_left_trim,  boxes_left,  idx_left,  chunk_px)
            (center_idx_right, center_xy_right) = foot_center_chunk(img_right_trim, boxes_right, idx_right, chunk_px)
            ext_left  = compute_extents(idx_left,  center_idx_left)
            ext_right = compute_extents(idx_right, center_idx_right)

            left_dict  = build_index_to_rect(idx_left,  boxes_left)
            right_dict = build_index_to_rect(idx_right, boxes_right)

            pairs = []
            if center_idx_left is not None and center_idx_right is not None and len(idx_left) > 0 and len(idx_right) > 0:
                right_set = set(idx_right)
                for l_idx in idx_left:
                    mapped = map_chunk_across_feet(
                        l_idx, center_idx_left, center_idx_right,
                        ext_left, ext_right, mirror=True
                    )
                    if mapped is None:
                        continue
                    if mapped not in right_set:
                        mapped = nearest_valid_index(mapped[0], mapped[1], idx_right)
                        if mapped is None:
                            continue
                    pairs.append((l_idx, mapped))

            suspicious_left  = set()
            suspicious_right = set()

            for (l_idx, r_idx) in pairs:
                if l_idx not in left_dict or r_idx not in right_dict:
                    continue
                lx0, ly0, lw, lh = left_dict[l_idx]
                rx0, ry0, rw, rh = right_dict[r_idx]
                l_region = img_left_trim[ly0:ly0+lh, lx0:lx0+lw]
                r_region = img_right_trim[ry0:ry0+rh, rx0:rx0+rw]
                l_mean = np.nanmean(l_region) if l_region.size > 0 else np.nan
                r_mean = np.nanmean(r_region) if r_region.size > 0 else np.nan
                if np.isnan(l_mean) or np.isnan(r_mean):
                    continue
                if abs(l_mean - r_mean) >= temp_diff_threshold:
                    suspicious_left.add(l_idx)
                    suspicious_right.add(r_idx)

            gap = np.full((img_left_trim.shape[0], gap_cols), np.nan)
            combined = np.hstack((img_left_trim, gap, img_right_trim))

            fig, ax = plt.subplots(figsize=figsize)
            im = ax.imshow(combined, cmap=colormap)
            ax.axis("off")
            ax.set_title(f"Day {di+1}: Wound detection (Δ ≥ {temp_diff_threshold} °C) — {chunk_px}×{chunk_px} px")

            left_w = img_left_trim.shape[1]
            for (x, y, w_box, h_box) in boxes_left:
                ax.add_patch(Rectangle((x, y), w_box, h_box, linewidth=0.7, edgecolor='black', facecolor='none'))
            x_offset_right = left_w + gap_cols
            for (x, y, w_box, h_box) in boxes_right:
                ax.add_patch(Rectangle((x + x_offset_right, y), w_box, h_box, linewidth=0.7, edgecolor='black', facecolor='none'))

            if center_xy_left is not None:
                cx0, cy0 = center_xy_left
                ax.add_patch(Rectangle((cx0, cy0), chunk_px, chunk_px, linewidth=1.0, edgecolor='blue', facecolor='blue', alpha=0.35))
            if center_xy_right is not None:
                cx0, cy0 = center_xy_right
                ax.add_patch(Rectangle((cx0 + x_offset_right, cy0), chunk_px, chunk_px, linewidth=1.0, edgecolor='blue', facecolor='blue', alpha=0.35))

            for l_idx in suspicious_left:
                if l_idx in left_dict:
                    x0, y0, _, _ = left_dict[l_idx]
                    ax.add_patch(Rectangle((x0, y0), chunk_px, chunk_px, linewidth=1.2, edgecolor='green', facecolor='green', alpha=0.35))
            for r_idx in suspicious_right:
                if r_idx in right_dict:
                    x0, y0, _, _ = right_dict[r_idx]
                    ax.add_patch(Rectangle((x0 + x_offset_right, y0), chunk_px, chunk_px, linewidth=1.2, edgecolor='green', facecolor='green', alpha=0.35))

            cbar = fig.colorbar(im, ax=ax, fraction=cbar_fraction, pad=cbar_pad)
            cbar.set_label("Temperature (°C)")

            # SINGLE SAVE: wound_detect_dayN.png into the separate output tree
            out_png = os.path.join(out_dir, f"wound_detect_day{di+1}.png")
            plt.savefig(out_png, dpi=300, bbox_inches="tight")
            plt.close()
            # === END: detection logic (unchanged) ===

        any_found = True

    if not any_found:
        print(f"No generated runs found under '{INPUT_ROOT}'. "
              f"Expected: {INPUT_ROOT}/<patient>/variant_xx/<mode>/mat/*.mat")

if __name__ == "__main__":
    main()
