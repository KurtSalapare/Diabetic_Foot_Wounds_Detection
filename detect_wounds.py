import os
import glob
import warnings
from collections import defaultdict, deque
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.io.matlab.miobase import MatReadWarning

# ==========================
# PIPELINE INTEGRATION
# ==========================
INPUT_ROOT = "output_images_wound_modes_segmented"
DETECT_OUTPUT_ROOT = "output_images_detect_wounds"

PATIENT_FILTER = None      # e.g. "gz7" or ["gz7","gz8"]; None = all discovered
VARIANT_FILTER = None      # e.g. "variant_01" or ["variant_01"]; None = all
RUNDIR_FILTER  = None      # e.g. "both_pre10_dev20_stat10"; None = all

# Which day indices (0-based). None = all days found.
days_to_process = None

# ==========================
# DETECTION CONFIG
# ==========================
chunk_px = 5
gap_cols = 5
temp_diff_threshold = 2.2

# Sliding-window & thresholds
WINDOW_DAYS = 10               # "10 (variable) day sliding window"
MIN_SUSPECT_TRIGGERS = 7       # ">=7 triggers in a 10-day window"
MIN_NEIGHBOR_SUSPECTS = 3      # ">=3 neighboring suspect cells"

# Visualization
figsize = (8, 6)
cbar_fraction = 0.035
cbar_pad = 0.04
colormap = "hot"

COLOR_TRIGGER = "green"        # per-day 2.2°C triggers (debug)
COLOR_SUSPECT = "blue"         # suspect cells (>=7 in window)
COLOR_WOUND   = (0.93, 0.0, 1) # bright violet/magenta
GRID_COLOR    = "black"

# Center-chunk debug control
SHOW_CENTER_DEBUG = False      # "initialize it to be off"
CENTER_FACE = "black"          # "change ... to a black fill instead"
CENTER_ALPHA = 0.40

# Make grid stable across days using the corrected canvases from the generator.
# If True, skip width-trimming so chunk indices remain consistent per run.
CONSISTENT_GRID = True

warnings.filterwarnings("ignore", category=MatReadWarning)

# Segmentation (match segment_foot): top 65%, heel 40% of that
SEG_TOP_FRAC   = 0.65   # top portion (heel + mid-foot)
SEG_HEEL_FRAC  = 0.40   # fraction of top portion that is heel

# ---------- Helpers (existing core logic kept and built upon) ----------

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

def _segment_boundaries(img_h):
    """
    Returns (heel_end_y, top_end_y) in pixel rows for a foot image of height img_h.
    heel: rows [0, heel_end_y)
    mid : rows [heel_end_y, top_end_y)
    upper: rows [top_end_y, img_h)
    """
    top_end = int(round(SEG_TOP_FRAC * img_h))
    heel_end = int(round(SEG_HEEL_FRAC * top_end))
    return heel_end, top_end

def _segment_name_from_y(y0, img_h):
    heel_end, top_end = _segment_boundaries(img_h)
    if y0 < heel_end:
        return "heel"
    elif y0 < top_end:
        return "mid-foot"
    else:
        return "upper-foot"

def _segments_for_side(suspects, rect_dict, img_h):
    """
    suspects: set of (ix,iy)
    rect_dict: {(ix,iy): (x0,y0,w,h)}
    """
    segs = set()
    for idx in suspects:
        if idx in rect_dict:
            _, y0, _, _ = rect_dict[idx]
            segs.add(_segment_name_from_y(y0, img_h))
    return segs

# ---------- NEW: I/O helpers (unchanged structure; now also logs/state) ----------

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

# ---------- NEW: sliding-window + clustering utilities ----------

def _update_deques(active_indices, today_triggers, hist_dict):
    """
    Ensure each active index has a deque, append 1/0 for today's trigger.
    """
    for idx in active_indices:
        if idx not in hist_dict:
            hist_dict[idx] = deque(maxlen=WINDOW_DAYS)
        hist_dict[idx].append(1 if idx in today_triggers else 0)

def _suspects_from_history(active_indices, hist_dict):
    suspects = set()
    for idx in active_indices:
        q = hist_dict.get(idx, None)
        if q is None:
            continue
        if sum(q) >= MIN_SUSPECT_TRIGGERS:
            suspects.add(idx)
    return suspects

def _components_from_suspects(suspects):
    """
    Find 4-neighbor connected components in the (ix,iy) grid index space.
    """
    suspects = set(suspects)
    comps = []
    visited = set()
    for node in suspects:
        if node in visited:
            continue
        stack = [node]
        comp = set()
        while stack:
            cur = stack.pop()
            if cur in visited or cur not in suspects:
                continue
            visited.add(cur)
            comp.add(cur)
            ix, iy = cur
            nbrs = [(ix+1,iy),(ix-1,iy),(ix,iy+1),(ix,iy-1)]
            for n in nbrs:
                if n in suspects and n not in visited:
                    stack.append(n)
        comps.append(comp)
    return comps

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

        use_idx = range(len(day_files)) if days_to_process is None else [
            i for i in days_to_process if 0 <= i < len(day_files)
        ]

        run_name = os.path.basename(run_path)
        out_png_dir = os.path.join(DETECT_OUTPUT_ROOT, patient_id, variant_dir, run_name, "png")
        out_log_dir = os.path.join(DETECT_OUTPUT_ROOT, patient_id, variant_dir, run_name, "log")
        out_state_dir = os.path.join(DETECT_OUTPUT_ROOT, patient_id, variant_dir, run_name, "state")
        os.makedirs(out_png_dir, exist_ok=True)
        os.makedirs(out_log_dir, exist_ok=True)
        os.makedirs(out_state_dir, exist_ok=True)

        client_log_path = os.path.join(out_log_dir, "client_log.txt")

        print(f"[detect] patient={patient_id} | {variant_dir} | run='{run_name}' | days={len(list(use_idx))} -> out='{os.path.relpath(out_png_dir)}'")

        # --- Per-run state (detection cycle) ---
        hist_left  = {}
        hist_right = {}
        detection_flag = False
        detection_side = None
        detection_day  = None
        detection_comp = None

        # Process each selected day
        for di in use_idx:
            day_mat = day_files[di]
            print(f"   → day {di + 1:02d}: {os.path.basename(day_mat)}")

            try:
                scan_left_raw, scan_right_raw, phase, progress = _load_day_mat(day_mat)
            except Exception as e:
                print(f"   ! skip {os.path.basename(day_mat)}: {e}")
                continue

            # Build display canvases (optionally skip trimming for stable grid)
            img_left  = make_display_image(scan_left_raw)
            img_right = make_display_image(scan_right_raw)

            if CONSISTENT_GRID:
                img_left_trim,  valid_cols_left  = img_left,  np.ones(img_left.shape[1], dtype=bool)
                img_right_trim, valid_cols_right = img_right, np.ones(img_right.shape[1], dtype=bool)
            else:
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

            # Map left->right chunk pairs
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

            # Collect today's per-pair means so we can tell which foot is hotter
            pair_stats = []  # list of (l_idx, r_idx, l_mean, r_mean)

            # ---- Per-day 2.2°C triggers (debug/primitive signal) ----
            triggers_left = set()
            triggers_right = set()

            for (l_idx, r_idx) in pairs:
                if l_idx not in left_dict or r_idx not in right_dict:
                    continue
                lx0, ly0, lw, lh = left_dict[l_idx]
                rx0, ry0, rw, rh = right_dict[r_idx]
                l_region = img_left_trim[ly0:ly0 + lh, lx0:lx0 + lw]
                r_region = img_right_trim[ry0:ry0 + rh, rx0:rx0 + rw]
                l_mean = np.nanmean(l_region) if l_region.size > 0 else np.nan
                r_mean = np.nanmean(r_region) if r_region.size > 0 else np.nan
                if np.isnan(l_mean) or np.isnan(r_mean):
                    continue

                # Save for directionality (which foot is hotter)
                pair_stats.append((l_idx, r_idx, float(l_mean), float(r_mean)))

                if abs(l_mean - r_mean) >= temp_diff_threshold:
                    triggers_left.add(l_idx)
                    triggers_right.add(r_idx)

            # ---- Sliding-window bookkeeping ----
            _update_deques(idx_left,  triggers_left,  hist_left)
            _update_deques(idx_right, triggers_right, hist_right)

            suspects_left  = _suspects_from_history(idx_left,  hist_left)
            suspects_right = _suspects_from_history(idx_right, hist_right)

            # Determine segments of suspect areas (per side)
            left_segs = _segments_for_side(suspects_left, left_dict, img_left_trim.shape[0])
            right_segs = _segments_for_side(suspects_right, right_dict, img_right_trim.shape[0])

            # Decide which foot is hotter today among suspect locations
            left_hot = 0
            right_hot = 0
            for (l_idx, r_idx, l_mean, r_mean) in pair_stats:
                if (l_idx in suspects_left) or (r_idx in suspects_right):
                    if l_mean > r_mean:
                        left_hot += 1
                    elif r_mean > l_mean:
                        right_hot += 1

            if left_hot > right_hot:
                hotter_side = "left"
            elif right_hot > left_hot:
                hotter_side = "right"
            else:
                # Tie-breaker: use the side with more suspects; if still tied, default to left
                if len(suspects_left) > len(suspects_right):
                    hotter_side = "left"
                elif len(suspects_right) > len(suspects_left):
                    hotter_side = "right"
                else:
                    hotter_side = "left"

            # Single-foot count and segments should reflect the hotter foot
            if hotter_side == "left":
                num_areas_one_foot = len(suspects_left)
                patient_segs = _segments_for_side(suspects_left, left_dict, img_left_trim.shape[0])
            else:
                num_areas_one_foot = len(suspects_right)
                patient_segs = _segments_for_side(suspects_right, right_dict, img_right_trim.shape[0])

            # Single-foot count (they should match, but use max defensively)
            num_areas_one_foot = max(len(suspects_left), len(suspects_right))

            # Which segments to surface to patient (favor left; fallback to right)
            patient_segs = left_segs or right_segs

            # ---- Clustering to detect wound ----
            comps_left  = [c for c in _components_from_suspects(suspects_left)  if len(c) >= MIN_NEIGHBOR_SUSPECTS]
            comps_right = [c for c in _components_from_suspects(suspects_right) if len(c) >= MIN_NEIGHBOR_SUSPECTS]

            # If any side has a qualifying cluster, mark detection and end cycle after drawing/saving today
            wound_clusters = []
            if len(comps_left)  > 0: wound_clusters.append(("left",  comps_left[0]))
            if len(comps_right) > 0: wound_clusters.append(("right", comps_right[0]))

            # ---- Visualization ----
            gap = np.full((img_left_trim.shape[0], gap_cols), np.nan)
            combined = np.hstack((img_left_trim, gap, img_right_trim))

            fig, ax = plt.subplots(figsize=figsize)
            im = ax.imshow(combined, cmap=colormap)
            ax.axis("off")
            ax.set_title(
                f"Day {di+1}: Sliding-window detection "
                f"(win={WINDOW_DAYS}, ≥{MIN_SUSPECT_TRIGGERS} triggers → suspect; "
                f"≥{MIN_NEIGHBOR_SUSPECTS} neighbors → wound)"
            )

            # Grid
            left_w = img_left_trim.shape[1]
            for (x, y, w_box, h_box) in boxes_left:
                ax.add_patch(Rectangle((x, y), w_box, h_box, linewidth=0.7, edgecolor=GRID_COLOR, facecolor='none'))
            x_offset_right = left_w + gap_cols
            for (x, y, w_box, h_box) in boxes_right:
                ax.add_patch(Rectangle((x + x_offset_right, y), w_box, h_box, linewidth=0.7, edgecolor=GRID_COLOR, facecolor='none'))

            # Optional center debug (now black fill)
            if SHOW_CENTER_DEBUG and (center_xy_left is not None):
                cx0, cy0 = center_xy_left
                ax.add_patch(Rectangle((cx0, cy0), chunk_px, chunk_px, linewidth=1.0,
                                       edgecolor=CENTER_FACE, facecolor=CENTER_FACE, alpha=CENTER_ALPHA))
            if SHOW_CENTER_DEBUG and (center_xy_right is not None):
                cx0, cy0 = center_xy_right
                ax.add_patch(Rectangle((cx0 + x_offset_right, cy0), chunk_px, chunk_px, linewidth=1.0,
                                       edgecolor=CENTER_FACE, facecolor=CENTER_FACE, alpha=CENTER_ALPHA))

            # Per-day 2.2°C triggers (green)
            for l_idx in triggers_left:
                if l_idx in left_dict:
                    x0, y0, _, _ = left_dict[l_idx]
                    ax.add_patch(Rectangle((x0, y0), chunk_px, chunk_px, linewidth=1.0,
                                           edgecolor=COLOR_TRIGGER, facecolor=COLOR_TRIGGER, alpha=0.28))
            for r_idx in triggers_right:
                if r_idx in right_dict:
                    x0, y0, _, _ = right_dict[r_idx]
                    ax.add_patch(Rectangle((x0 + x_offset_right, y0), chunk_px, chunk_px, linewidth=1.0,
                                           edgecolor=COLOR_TRIGGER, facecolor=COLOR_TRIGGER, alpha=0.28))

            # Suspect cells (blue)
            for l_idx in suspects_left:
                if l_idx in left_dict:
                    x0, y0, _, _ = left_dict[l_idx]
                    ax.add_patch(Rectangle((x0, y0), chunk_px, chunk_px, linewidth=1.2,
                                           edgecolor=COLOR_SUSPECT, facecolor=COLOR_SUSPECT, alpha=0.45))
            for r_idx in suspects_right:
                if r_idx in right_dict:
                    x0, y0, _, _ = right_dict[r_idx]
                    ax.add_patch(Rectangle((x0 + x_offset_right, y0), chunk_px, chunk_px, linewidth=1.2,
                                           edgecolor=COLOR_SUSPECT, facecolor=COLOR_SUSPECT, alpha=0.45))

            # Wound clusters (bright violet) — draw last so they stand out
            if wound_clusters:
                for side, comp in wound_clusters:
                    for idx in comp:
                        if side == "left" and idx in left_dict:
                            x0, y0, _, _ = left_dict[idx]
                            ax.add_patch(Rectangle((x0, y0), chunk_px, chunk_px, linewidth=1.3,
                                                   edgecolor=COLOR_WOUND, facecolor=COLOR_WOUND, alpha=0.65))
                        if side == "right" and idx in right_dict:
                            x0, y0, _, _ = right_dict[idx]
                            ax.add_patch(Rectangle((x0 + x_offset_right, y0), chunk_px, chunk_px, linewidth=1.3,
                                                   edgecolor=COLOR_WOUND, facecolor=COLOR_WOUND, alpha=0.65))

            cbar = fig.colorbar(im, ax=ax, fraction=cbar_fraction, pad=cbar_pad)
            cbar.set_label("Temperature (°C)")

            out_png = os.path.join(out_png_dir, f"wound_detect_day{di+1}.png")
            plt.savefig(out_png, dpi=300, bbox_inches="tight")
            plt.close()

            # ---- Client-friendly daily log line ----
            left_sus_n = len(suspects_left)
            right_sus_n = len(suspects_right)

            if wound_clusters:
                # Earliest qualifying cluster for messaging
                detection_side, detection_comp = wound_clusters[0]
                detection_flag, detection_day = True, di + 1

                # Segment(s) for the detected wound cluster (on that side)
                if detection_side == "left":
                    wound_segs = _segments_for_side(detection_comp, left_dict, img_left_trim.shape[0])
                else:
                    wound_segs = _segments_for_side(detection_comp, right_dict, img_right_trim.shape[0])

                seg_txt = ", ".join(sorted(wound_segs)) if wound_segs else "unspecified area"
                msg = (
                    f"Day {di + 1:02d}: Suspect wound detected on {detection_side} foot "
                    f"({len(detection_comp)} contiguous hot areas) at {seg_txt}."
                )
                print("   !", msg)


            elif (left_sus_n + right_sus_n) > 0:
                seg_txt = ", ".join(sorted(patient_segs)) if patient_segs else "unspecified area"
                msg = (
                    f"Day {di + 1:02d}: Elevated warmth noted — {num_areas_one_foot} area(s) "
                    f"on the {hotter_side} foot at {seg_txt}. Monitoring continues."
                )
            else:
                msg = f"Day {di + 1:02d}: No anomaly detected."

            with open(client_log_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")

            # End cycle if wound detected
            if detection_flag:
                print(f"[detect] WOUND DETECTED — patient={patient_id}, {variant_dir}, run='{run_name}', day={detection_day:02d}. Cycle ended.")
                break

        # Save per-run trigger history snapshots (for debugging/analytics)
        def _hist_to_array(hist_dict):
            arr = []
            for (ix,iy), q in hist_dict.items():
                arr.append((ix, iy, int(sum(q)), list(q)))
            return np.array(arr, dtype=object)

        np.savez(
            os.path.join(out_state_dir, "trigger_counters.npz"),
            left=_hist_to_array(hist_left),
            right=_hist_to_array(hist_right),
            window_days=WINDOW_DAYS,
            min_suspect_triggers=MIN_SUSPECT_TRIGGERS,
            min_neighbor_suspects=MIN_NEIGHBOR_SUSPECTS
        )

        any_found = True

    if not any_found:
        print(f"No generated runs found under '{INPUT_ROOT}'. Expected: {INPUT_ROOT}/<patient>/variant_xx/<mode>/mat/*.mat")

if __name__ == "__main__":
    main()
