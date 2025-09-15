import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
from scipy.io.matlab import MatReadWarning
from scipy.ndimage import gaussian_filter

# ==========================
# CONFIGURATION SECTION
# ==========================

# Input / output
mat_file = "./Data/Temp Data/gz1.mat"   # MAT file to read from
output_dir = "output_images_with_wound"

# Which day to use
day_index = 0   # 0 = Day 1, 1 = Day 2, etc.

# Which foot to apply wound on
# Options: "left" or "right"
apply_to = "left"

# Wound positioning mode
# 1 = center of foot
# 2 = random within foot
# 3 = manual coordinate
position_mode = 2
manual_coord = (150, 180)  # (y, x) only used if position_mode == 3

# Wound parameters
core_radii   = (20, 20)    # ellipse radii for core (a, b)
inflam_radii = (40, 40)    # ellipse radii for inflammation (a, b)
core_temp    = 1.5         # core "heat" intensity
inflam_temp  = 0.5         # inflammation "heat" intensity

# Gaussian blur for realism
blur_sigma_core   = 3.0
blur_sigma_inflam = 6.0

# ==========================
# HELPER FUNCTIONS SECTION
# ==========================

def trim_empty_columns(img):
    """Remove fully-zero/NaN columns to bring feet closer."""
    valid_cols = ~(np.all(np.isnan(img) | (img == 0), axis=0))
    return img[:, valid_cols]

# ==========================
# MAIN CODE
# ==========================

if __name__ == "__main__":

    # Silence MAT-file warning
    warnings.filterwarnings("ignore", category=MatReadWarning)

    # Load .mat file
    mat = scipy.io.loadmat(mat_file)

    # NOTE: swapped assignment (your setup)
    left_crop  = mat["Direct_plantar_Right_crop"]
    right_crop = mat["Direct_plantar_Left_crop"]

    # Select chosen day
    scan_left  = left_crop[day_index, 0]
    scan_right = right_crop[day_index, 0]

    img_left  = np.where(scan_left == 0, np.nan, scan_left)
    img_right = np.where(scan_right == 0, np.nan, scan_right)

    # Decide which foot to apply to
    target_img = img_left if apply_to == "left" else img_right
    h, w = target_img.shape

    # ==========================
    # Find wound center
    # ==========================
    ys, xs = np.where(~np.isnan(target_img))  # foot pixels only

    if position_mode == 1:  # center of foot
        y_center = int(np.mean(ys))
        x_center = int(np.mean(xs))
    elif position_mode == 2:  # random inside foot
        idx = np.random.randint(len(ys))
        y_center, x_center = ys[idx], xs[idx]
    elif position_mode == 3:  # manual
        y_center, x_center = manual_coord
    else:
        raise ValueError("Invalid position_mode. Use 1, 2, or 3.")

    # ==========================
    # Generate wound mask
    # ==========================
    Y, X = np.ogrid[:h, :w]

    core_mask = ((X - x_center)**2 / core_radii[0]**2 +
                (Y - y_center)**2 / core_radii[1]**2) <= 1
    inflam_mask = ((X - x_center)**2 / inflam_radii[0]**2 +
                (Y - y_center)**2 / inflam_radii[1]**2) <= 1

    mask_core   = np.zeros((h, w))
    mask_inflam = np.zeros((h, w))

    mask_core[core_mask]     = core_temp
    mask_inflam[inflam_mask] = inflam_temp

    # Apply Gaussian blur
    mask_core   = gaussian_filter(mask_core, sigma=blur_sigma_core)
    mask_inflam = gaussian_filter(mask_inflam, sigma=blur_sigma_inflam)

    # Combine masks (core overwrites inflammation at overlap)
    mask = np.maximum(mask_inflam, mask_core)

    # Save mask separately
    os.makedirs(output_dir, exist_ok=True)
    scipy.io.savemat(os.path.join(output_dir, f"wound_mask_day{day_index+1}.mat"),
                    {"wound_mask": mask})

    # ==========================
    # Apply mask to chosen foot
    # ==========================
    if apply_to == "left":
        img_left_with_wound  = np.where(~np.isnan(img_left), img_left + mask, np.nan)
        img_right_with_wound = img_right
    else:
        img_left_with_wound  = img_left
        img_right_with_wound = np.where(~np.isnan(img_right), img_right + mask, np.nan)

    # ==========================
    # Trim & combine (as in process_mat_files.py)
    # ==========================
    img_left_trim  = trim_empty_columns(img_left_with_wound)
    img_right_trim = trim_empty_columns(img_right_with_wound)

    gap = np.full((img_left_trim.shape[0], 5), np.nan)
    combined = np.hstack((img_left_trim, gap, img_right_trim))

    # ==========================
    # Plot and save
    # ==========================
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(combined, cmap="hot")
    ax.axis("off")
    ax.set_title(f"Day {day_index+1}: Wound on {apply_to}")

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
    cbar.set_label("Temperature (°C)")

    plt.savefig(os.path.join(output_dir, f"foot_day{day_index+1}_with_wound.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved combined PNG + MAT in '{output_dir}' (wound on {apply_to} foot)")