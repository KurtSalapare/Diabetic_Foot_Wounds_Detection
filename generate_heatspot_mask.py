import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
from scipy.io.matlab import MatReadWarning
from scipy.ndimage import gaussian_filter
from skimage.morphology import disk
from skimage.transform import rescale

# ==========================
# CONFIGURATION SECTION
# ==========================

mat_file = "Data/Temp Data/gz1.mat"
output_dir = "output_images_wound_modes"

apply_to = "left"                # "left" or "right"
generation_mode = "developing"          # "static", "developing", "both"
develop_mode = "size+intensity"   # "size+intensity" or "intensity-only"

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

warnings.filterwarnings("ignore", category=MatReadWarning)

mat = scipy.io.loadmat(mat_file)
left_crop  = mat["Direct_plantar_Right_crop"]
right_crop = mat["Direct_plantar_Left_crop"]

def trim_empty_columns(img):
    valid_cols = ~(np.all(np.isnan(img) | (img == 0), axis=0))
    return img[:, valid_cols]

os.makedirs(output_dir, exist_ok=True)
num_days = left_crop.shape[0]

# --------------------------
# 1) Pick center and shape
# --------------------------
scan_left  = left_crop[0, 0]
scan_right = right_crop[0, 0]
img_left  = np.where(scan_left == 0, np.nan, scan_left)
img_right = np.where(scan_right == 0, np.nan, scan_right)

target_img = img_left if apply_to == "left" else img_right
h, w = target_img.shape

ys, xs = np.where(~np.isnan(target_img))
if position_mode == 1:
    y_center = int(np.mean(ys)); x_center = int(np.mean(xs))
elif position_mode == 2:
    idx = np.random.randint(len(ys))
    y_center, x_center = int(ys[idx]), int(xs[idx])
elif position_mode == 3:
    y_center, x_center = manual_coord
else:
    raise ValueError("Invalid position_mode")

shape_mode = np.random.choice(["circle", "multi"])
print(f"Wound generated at ({y_center},{x_center}) with shape '{shape_mode}'")

Y, X = np.ogrid[:h, :w]

# --------------------------
# 2) Build final mask (locked shape)
# --------------------------
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
# 3) Loop through days
# --------------------------
modes_to_run = [generation_mode] if generation_mode in ["static","developing"] else ["static","developing"]

for mode in modes_to_run:
    for i in range(num_days):
        progress = (i / (num_days - 1)) if (num_days > 1) else 1.0
        mask_day = make_mask(progress if mode == "developing" else 1.0)

        scipy.io.savemat(os.path.join(output_dir, f"wound_mask_day{i+1}_{mode}.mat"),
                         {"wound_mask": mask_day})

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

        plt.savefig(os.path.join(output_dir, f"foot_day{i+1}_{mode}.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()
        
        scipy.io.savemat(os.path.join(output_dir, f"wound_mask+foot_day{i+1}_{mode}.mat"),
                         {"left_foot": img_left_trim, "right_foot": img_right_trim})

print(f"Saved images and masks for modes {modes_to_run} in '{output_dir}'")
