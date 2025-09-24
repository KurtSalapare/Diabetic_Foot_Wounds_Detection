import os
import warnings
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.io.matlab.miobase import MatReadWarning

# ==========================
# CONFIGURATION SECTION
# ==========================

mat_file = "Data/Temp Data/gz14.mat"  # input .mat (INDIRECT crops)
output_dir = "output_wound_detection" # output folder

# Process which days (None = all)
days_to_process = None  # e.g., [0,1,2] for first 3 days

# Pixel-based chunk size (square)
chunk_px = 5  # e.g., 5x5 pixels per chunk

# Visual gap between feet in combined image (columns of NaN)
gap_cols = 5

# Detection threshold (°C)
temp_diff_threshold = 2.2

# Figure / colorbar styling (keep consistent with previous)
figsize = (8, 6)
cbar_fraction = 0.035
cbar_pad = 0.04
colormap = "hot"

# ==========================
# END CONFIG
# ==========================

warnings.filterwarnings("ignore", category=MatReadWarning)

# Load .mat with INDIRECT crops (as requested)
mat = scipy.io.loadmat(mat_file)
left_crop  = mat["Indirect_plantar_Right_crop"]  # per instruction
right_crop = mat["Indirect_plantar_Left_crop"]   # per instruction

# ---------- Helpers ----------

def trim_empty_columns(img):
    """Remove fully-zero/NaN columns to bring feet closer; return trimmed image + valid cols mask."""
    valid_cols = ~(np.all(np.isnan(img) | (img == 0), axis=0))
    return img[:, valid_cols], valid_cols

def make_display_image(scan_raw):
    """Map 0 -> NaN so background is invisible in imshow."""
    return np.where(scan_raw == 0, np.nan, scan_raw)

def compute_valid_grid_boxes(scan_raw, img_trimmed, valid_cols_bool, chunk_px):
    """
    Build grid boxes in TRIMMED coordinates and their (ix, iy) indices.
    Rules:
      - Only chunks fully inside foot (no NaNs in img_trimmed region).
      - Exclude if ANY 0 in underlying RAW region (trimmed).
    """
    scan_raw_trimmed = scan_raw[:, valid_cols_bool]
    h, w = img_trimmed.shape
    boxes = []           # list of (x0, y0, w, h) in trimmed coords
    indices = []         # list of (ix, iy) chunk indices (x0//chunk_px, y0//chunk_px)

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
    """
    Find the valid chunk that contains the centroid of the foot (in TRIMMED coords).
    If that chunk isn't valid, pick the nearest valid chunk.
    Returns: (center_idx=(icx, icy), center_xy=(x0, y0)) or (None, None)
    """
    ys, xs = np.where(~np.isnan(img_trimmed))
    if len(xs) == 0:
        return None, None
    cx = np.mean(xs)
    cy = np.mean(ys)

    icx = int(cx // chunk_px)
    icy = int(cy // chunk_px)
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
    """
    Per-foot max extents in chunk-index space relative to center.
    Returns: (max_abs_dx, max_abs_dy), each at least 1.
    """
    if center_idx is None or len(indices) == 0:
        return 1, 1
    icx, icy = center_idx
    dx = [ix - icx for ix, iy in indices]
    dy = [iy - icy for ix, iy in indices]
    max_abs_dx = max(1, int(np.max(np.abs(dx))))
    max_abs_dy = max(1, int(np.max(np.abs(dy))))
    return max_abs_dx, max_abs_dy

def nearest_valid_index(target_ix, target_iy, indices):
    """
    If the mapped index is invalid, snap to the nearest valid index (Euclidean, in index space).
    """
    if len(indices) == 0:
        return None
    idx_arr = np.array(indices)
    d2 = (idx_arr[:,0] - target_ix)**2 + (idx_arr[:,1] - target_iy)**2
    k = int(np.argmin(d2))
    return tuple(idx_arr[k])

def map_chunk_across_feet(src_idx, src_center_idx, dst_center_idx,
                          src_extents, dst_extents, mirror=True):
    """
    Map a source chunk index (ix, iy) from one foot to the other using:
      - reference centers (per-foot),
      - per-foot extents (max |dx|, |dy| from center),
      - optional horizontal mirroring.
    """
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
    """Dict: (ix, iy) -> (x0, y0, w, h) in TRIMMED coords."""
    d = {}
    for (x0, y0, w0, h0), (ix0, iy0) in zip(boxes, indices):
        d[(ix0, iy0)] = (x0, y0, w0, h0)
    return d

# ---------- Main per-day loop ----------

os.makedirs(output_dir, exist_ok=True)

num_days = left_crop.shape[0]
day_indices = range(num_days) if days_to_process is None else days_to_process

for i in day_indices:
    # raw scans
    scan_left_raw  = left_crop[i, 0]
    scan_right_raw = right_crop[i, 0]

    # display images
    img_left  = make_display_image(scan_left_raw)
    img_right = make_display_image(scan_right_raw)

    # trim columns independently
    img_left_trim,  valid_cols_left  = trim_empty_columns(img_left)
    img_right_trim, valid_cols_right = trim_empty_columns(img_right)

    # build grids (valid chunks only) per foot, in TRIMMED coords
    boxes_left,  idx_left  = compute_valid_grid_boxes(scan_left_raw,  img_left_trim,  valid_cols_left,  chunk_px)
    boxes_right, idx_right = compute_valid_grid_boxes(scan_right_raw, img_right_trim, valid_cols_right, chunk_px)

    # find center chunk per foot (reference) and extents
    (center_idx_left,  center_xy_left)  = foot_center_chunk(img_left_trim,  boxes_left,  idx_left,  chunk_px)
    (center_idx_right, center_xy_right) = foot_center_chunk(img_right_trim, boxes_right, idx_right, chunk_px)
    ext_left  = compute_extents(idx_left,  center_idx_left)
    ext_right = compute_extents(idx_right, center_idx_right)

    # fast lookups
    left_dict  = build_index_to_rect(idx_left,  boxes_left)
    right_dict = build_index_to_rect(idx_right, boxes_right)

    # map LEFT -> RIGHT to form mirrored pairs
    pairs = []  # list of ((l_ix,l_iy), (r_ix,r_iy))
    if center_idx_left is not None and center_idx_right is not None and len(idx_left) > 0 and len(idx_right) > 0:
        right_set = set(idx_right)
        for l_idx in idx_left:
            mapped = map_chunk_across_feet(
                l_idx, center_idx_left, center_idx_right,
                ext_left, ext_right, mirror=True
            )
            if mapped is None:
                continue
            # snap to nearest valid if exact mapped idx is invalid
            if mapped not in right_set:
                mapped = nearest_valid_index(mapped[0], mapped[1], idx_right)
                if mapped is None:
                    continue
            pairs.append((l_idx, mapped))

    # --- compute suspicion per pair ---
    suspicious_left  = set()
    suspicious_right = set()

    # Prepare trimmed RAW arrays for stats (not strictly needed since we already excluded zeros,
    # but we use img_trim with NaNs to compute means safely).
    for (l_idx, r_idx) in pairs:
        # locate rectangles (x0,y0,w,h) in trimmed coords
        if l_idx not in left_dict or r_idx not in right_dict:
            continue
        lx0, ly0, lw, lh = left_dict[l_idx]
        rx0, ry0, rw, rh = right_dict[r_idx]

        # extract chunk regions from trimmed display arrays (NaNs mark background)
        l_region = img_left_trim[ly0:ly0+lh, lx0:lx0+lw]
        r_region = img_right_trim[ry0:ry0+rh, rx0:rx0+rw]

        # compute mean temperatures per chunk (ignore NaNs)
        l_mean = np.nanmean(l_region) if l_region.size > 0 else np.nan
        r_mean = np.nanmean(r_region) if r_region.size > 0 else np.nan
        if np.isnan(l_mean) or np.isnan(r_mean):
            continue

        # suspicious if absolute mean difference >= threshold
        if abs(l_mean - r_mean) >= temp_diff_threshold:
            suspicious_left.add(l_idx)
            suspicious_right.add(r_idx)

    # ------------- Combine feet and draw -------------
    gap = np.full((img_left_trim.shape[0], gap_cols), np.nan)
    combined = np.hstack((img_left_trim, gap, img_right_trim))

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(combined, cmap=colormap)
    ax.axis("off")
    ax.set_title(f"Day {i+1}: Wound detection (Δ ≥ {temp_diff_threshold} °C) — {chunk_px}×{chunk_px} px")

    # draw black grid
    left_w = img_left_trim.shape[1]
    for (x, y, w_box, h_box) in boxes_left:
        ax.add_patch(Rectangle((x, y), w_box, h_box, linewidth=0.7, edgecolor='black', facecolor='none'))
    x_offset_right = left_w + gap_cols
    for (x, y, w_box, h_box) in boxes_right:
        ax.add_patch(Rectangle((x + x_offset_right, y), w_box, h_box, linewidth=0.7, edgecolor='black', facecolor='none'))

    # highlight center chunks BLUE
    if center_xy_left is not None:
        cx0, cy0 = center_xy_left
        ax.add_patch(Rectangle((cx0, cy0), chunk_px, chunk_px,
                               linewidth=1.0, edgecolor='blue', facecolor='blue', alpha=0.35))
    if center_xy_right is not None:
        cx0, cy0 = center_xy_right
        ax.add_patch(Rectangle((cx0 + x_offset_right, cy0), chunk_px, chunk_px,
                               linewidth=1.0, edgecolor='blue', facecolor='blue', alpha=0.35))

    # highlight suspicious chunks GREEN (both feet)
    for l_idx in suspicious_left:
        if l_idx in left_dict:
            x0, y0, _, _ = left_dict[l_idx]
            ax.add_patch(Rectangle((x0, y0), chunk_px, chunk_px,
                                   linewidth=1.2, edgecolor='green', facecolor='green', alpha=0.35))
    for r_idx in suspicious_right:
        if r_idx in right_dict:
            x0, y0, _, _ = right_dict[r_idx]
            ax.add_patch(Rectangle((x0 + x_offset_right, y0), chunk_px, chunk_px,
                                   linewidth=1.2, edgecolor='green', facecolor='green', alpha=0.35))

    # colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=cbar_fraction, pad=cbar_pad)
    cbar.set_label("Temperature (°C)")

    # save
    os.makedirs(output_dir, exist_ok=True)
    out_png = os.path.join(output_dir, f"wound_detect_day{i+1}.png")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

print(f"Saved wound-detection PNGs in '{output_dir}'")
