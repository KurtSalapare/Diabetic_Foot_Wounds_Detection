import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
from scipy.io.matlab import MatReadWarning
from scipy.ndimage import gaussian_filter
from skimage.morphology import disk
from skimage.transform import rescale

from foot_part_identifier import horizontal_split_by_percentage, vertical_split_by_percentage, segment_foot

# ==========================
# CONFIGURATION SECTION
# ==========================

mat_file = "Data/Temp Data/gz9.mat"
output_dir = "output_images_wound_modes"

apply_to = "right"                # "left" or "right"
generation_mode = "developing"          # "static", "developing", "both"
develop_mode = "size+intensity"   # "size+intensity" or "intensity-only"
apply_wound_to = "upper"             # "heel" or "upper_foot"

position_mode = 2                 # 1=center, 2=random, 3=manual
manual_coord = (150, 180)

# final wound parameters
core_radius_final   = 15
inflam_radius_final = 30
core_temp_final   = 5.0
inflam_temp_final = 3.0

# Gaussian blur
blur_sigma_core   = 6.0
blur_sigma_inflam = 6.0

# multi-blob config
multi_min_blobs = 2
multi_max_blobs = 6

# development start scales
initial_size_scale = 0.05
initial_temp_scale = 0.05

# ==========================

# --------------------------
# Helper: Build final mask (locked shape)
# --------------------------
def build_final_mask(shape_mode, x_center, y_center, core_radius_final, inflam_radius_final, multi_min_blobs, multi_max_blobs):
    final_core_mask, final_inflam_mask = 0, 0
    
    if shape_mode == "circle":
        final_core_mask   = (X - x_center)**2 + (Y - y_center)**2 <= core_radius_final**2
        final_inflam_mask = (X - x_center)**2 + (Y - y_center)**2 <= inflam_radius_final**2

    elif shape_mode == "multi":
        n_blobs = np.random.randint(multi_min_blobs, multi_max_blobs + 1)
        core_mask_bin = np.zeros((h, w), dtype=bool)
        inflam_mask_bin = np.zeros((h, w), dtype=bool)

        centers = [(x_center, y_center)]
        core_r_list = [core_radius_final]
        inflam_r_list = [inflam_radius_final]

        for _ in range(n_blobs - 1):
            base_x, base_y = centers[np.random.randint(len(centers))]
            angle = np.random.uniform(0, 2*np.pi)
            dist = np.random.randint(core_radius_final//2, core_radius_final*2)
            dx, dy = int(np.cos(angle) * dist), int(np.sin(angle) * dist)
            new_x = np.clip(base_x + dx, 0, w-1)
            new_y = np.clip(base_y + dy, 0, h-1)
            centers.append((new_x, new_y))
            core_r_list.append(np.random.randint(core_radius_final//2, core_radius_final))
            inflam_r_list.append(np.random.randint(inflam_radius_final//2, inflam_radius_final))

        for (cx, cy), rc, ri in zip(centers, core_r_list, inflam_r_list):
            core_mask_bin   |= (X - cx)**2 + (Y - cy)**2 <= rc**2
            inflam_mask_bin |= (X - cx)**2 + (Y - cy)**2 <= ri**2
            
        final_core_mask = core_mask_bin
        final_inflam_mask = inflam_mask_bin
    
    return final_core_mask, final_inflam_mask

# --------------------------
# Helper: scale whole union mask for developing mode
# --------------------------
def scale_mask(mask, scale_factor):
    """Scale a binary mask around its centroid."""
    # compute centroid of the mask
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return mask
    cx, cy = np.mean(xs), np.mean(ys)

    # recenter coordinates
    shift_x, shift_y = w//2 - cx, h//2 - cy
    shifted = np.roll(mask, (int(shift_y), int(shift_x)), axis=(0, 1))

    # scale around center
    resized = rescale(shifted.astype(float),
                      scale=scale_factor,
                      preserve_range=True,
                      anti_aliasing=True,
                      order=1)

    # pad/crop back to original size
    rh, rw = resized.shape
    out = np.zeros_like(mask, dtype=bool)
    sy = (out.shape[0] - rh)//2
    sx = (out.shape[1] - rw)//2
    sy = max(sy, 0); sx = max(sx, 0)
    ey = min(out.shape[0], sy+rh)
    ex = min(out.shape[1], sx+rw)
    out[sy:ey, sx:ex] = resized[:ey-sy, :ex-sx] > 0.5

    # shift back
    final = np.roll(out, (-int(shift_y), -int(shift_x)), axis=(0, 1))
    return final

# --------------------------
# Helper: make mask for a given day
# --------------------------
def make_mask(progress):
    if generation_mode == "static":
        core_val = core_temp_final
        inflam_val = inflam_temp_final
        core_mask = final_core_mask
        inflam_mask = final_inflam_mask

    else:  # developing
        if develop_mode == "size+intensity":
            scale_factor = initial_size_scale + (1.0 - initial_size_scale) * progress
            core_mask = scale_mask(final_core_mask, scale_factor)
            inflam_mask = scale_mask(final_inflam_mask, scale_factor)
            core_val = core_temp_final * (initial_temp_scale + (1 - initial_temp_scale) * progress)
            inflam_val = inflam_temp_final * (initial_temp_scale + (1 - initial_temp_scale) * progress)

        elif develop_mode == "intensity-only":
            core_mask = final_core_mask
            inflam_mask = final_inflam_mask
            core_val = core_temp_final * (initial_temp_scale + (1 - initial_temp_scale) * progress)
            inflam_val = inflam_temp_final * (initial_temp_scale + (1 - initial_temp_scale) * progress)

    mask_core = np.zeros((h, w)); mask_inflam = np.zeros((h, w))
    mask_core[core_mask] = core_val
    mask_inflam[inflam_mask] = inflam_val

    mask_core = gaussian_filter(mask_core, sigma=blur_sigma_core)
    mask_inflam = gaussian_filter(mask_inflam, sigma=blur_sigma_inflam)
    return np.maximum(mask_inflam, mask_core)

# --------------------------
# Helper: Trims Empty columns away
# --------------------------
def trim_empty_columns(img):
        valid_cols = ~(np.all(np.isnan(img) | (img == 0), axis=0))
        return img[:, valid_cols]
    
# --------------------------
# Helper: Pick center and shape
# --------------------------
def select_center_and_shape(left_crop, right_crop):
    scan_left  = left_crop[0, 0]
    scan_right = right_crop[0, 0]
    img_left  = np.where(scan_left == 0, np.nan, scan_left)
    img_right = np.where(scan_right == 0, np.nan, scan_right)

    target_foot = img_left if apply_to == "left" else img_right
    heel, mid_foot, upper_foot = segment_foot(target_foot)
    h, w = target_foot.shape
    
    if (apply_wound_to == "heel"):
        upper_and_mid_foot = np.vstack((mid_foot, upper_foot))
        upper_and_mid_foot[~np.isnan(upper_and_mid_foot)] = np.nan
        target_area_on_foot = np.vstack((heel, upper_and_mid_foot))
    else:
        heel_and_mid_foot = np.vstack((mid_foot, upper_foot))
        heel_and_mid_foot[~np.isnan(heel_and_mid_foot)] = np.nan
        target_area_on_foot = np.vstack((heel_and_mid_foot, upper_foot))

    ys, xs = np.where(~np.isnan(target_area_on_foot))
    if position_mode == 1:
        y_center = int(np.mean(ys)); x_center = int(np.mean(xs))
    elif position_mode == 2:
        idx = np.random.randint(len(ys))
        y_center, x_center = ys[idx], xs[idx]
    elif position_mode == 3:
        y_center, x_center = manual_coord
    else:
        raise ValueError("Invalid position_mode")

    shape_mode = np.random.choice(["circle", "multi"])
    
    print(f"Wound generated at ({y_center},{x_center}) with shape '{shape_mode}'")
    return y_center, x_center, shape_mode, h, w

def simulate_wound_gen(modes_to_run, num_days, left_crop, right_crop):
    for mode in modes_to_run:
        for i in range(num_days):
            progress = (i / (num_days - 1)) if (num_days > 1) else 1.0
            mask_day = make_mask(progress if mode == "developing" else 1.0)

            # scipy.io.savemat(os.path.join(output_dir, f"wound_mask_day{i+1}_{mode}.mat"),
            #                  {"wound_mask": mask_day})

            scan_left  = left_crop[i, 0]
            scan_right = right_crop[i, 0]
            img_left  = np.where(scan_left == 0, np.nan, scan_left)
            img_right = np.where(scan_right == 0, np.nan, scan_right)

            if apply_to == "left":
                img_left_with_wound  = np.where(~np.isnan(img_left), img_left + mask_day, np.nan)
                img_right_with_wound = img_right
            else:
                img_left_with_wound  = img_left
                img_right_with_wound = np.where(~np.isnan(img_right), img_right + mask_day, np.nan)

            img_left_trim  = trim_empty_columns(img_left_with_wound)
            img_right_trim = trim_empty_columns(img_right_with_wound)
            gap = np.full((img_left_trim.shape[0], 5), np.nan)
            combined = np.hstack((img_left_trim, gap, img_right_trim))

            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(combined, cmap="hot")
            ax.axis("off")
            ax.set_title(f"Day {i+1} ({mode}) - Wound on {apply_to}")

            cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
            cbar.set_label("Temperature (°C)")

            plt.show()

            # plt.savefig(os.path.join(output_dir, f"foot_day{i+1}_{mode}.png"),
            #             dpi=300, bbox_inches="tight")
            # plt.close()
            
            # scipy.io.savemat(os.path.join(output_dir, f"wound_mask+foot_day{i+1}_{mode}.mat"),
            #                  {"left_foot": img_left_trim, "right_foot": img_right_trim})

    print(f"Saved images and masks for modes {modes_to_run} in '{output_dir}'")

# --------------------------------------------
# MAIN FUNCTION
# --------------------------------------------

if __name__ == "__main__":
    
    warnings.filterwarnings("ignore", category=MatReadWarning)

    mat = scipy.io.loadmat(mat_file)
    left_crop  = mat["Indirect_plantar_Right_crop"]
    right_crop = mat["Indirect_plantar_Left_crop"]

    os.makedirs(output_dir, exist_ok=True)
    num_days = left_crop.shape[0]
    
    # --------------------------
    # 1) Pick center and shape
    # --------------------------
    y_center, x_center, shape_mode, h, w = select_center_and_shape(left_crop, right_crop)

    Y, X = np.ogrid[:h, :w]

    # --------------------------
    # 2) User Helper to 
    # Build final mask 
    # (locked shape)
    # --------------------------
    final_core_mask, final_inflam_mask = build_final_mask(shape_mode, x_center, y_center, core_radius_final, inflam_radius_final, multi_min_blobs, multi_max_blobs)

    # --------------------------
    # 3) Loop through days
    # --------------------------
    modes_to_run = [generation_mode] if generation_mode in ["static","developing"] else ["developing","static"]
    
    simulate_wound_gen(modes_to_run, num_days, left_crop, right_crop)
