import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
from scipy.io.matlab.miobase import MatReadWarning
from scipy.ndimage import gaussian_filter

# ==========================
# CONFIGURATION SECTION
# ==========================

mat_file = "Data/Temp Data/gz1.mat"   # MAT file
output_dir = "output_images_with_wounds"

# Which foot to apply wound on ("left" or "right")
apply_to = "left"

# Wound positioning mode:
# 1 = center of foot
# 2 = random within foot
# 3 = manual coordinate
position_mode = 2
manual_coord = (150, 180)  # only used if position_mode == 3

# Wound parameters
core_radii   = (15, 15)    # ellipse radii for core (a, b)
inflam_radii = (30, 30)    # ellipse radii for inflammation (a, b)
core_temp    = 3.0         # core "heat" intensity
inflam_temp  = 1.5         # inflammation "heat" intensity

# Gaussian blur for realism
blur_sigma_core   = 6.0
blur_sigma_inflam = 5.0

# ==========================
# MAIN CODE
# ==========================

warnings.filterwarnings("ignore", category=MatReadWarning)

# Load .mat file
mat = scipy.io.loadmat(mat_file)

# NOTE: swapped assignment (your setup)
left_crop  = mat["Direct_plantar_Right_crop"]
right_crop = mat["Direct_plantar_Left_crop"]

def trim_empty_columns(img):
    """Remove fully-zero/NaN columns to bring feet closer."""
    valid_cols = ~(np.all(np.isnan(img) | (img == 0), axis=0))
    return img[:, valid_cols]

# Create output directory
os.makedirs(output_dir, exist_ok=True)

num_days = left_crop.shape[0]

# ==========================
# Generate wound ONCE (locked across all days)
# ==========================

scan_left  = left_crop[0, 0]
scan_right = right_crop[0, 0]

img_left  = np.where(scan_left == 0, np.nan, scan_left)
img_right = np.where(scan_right == 0, np.nan, scan_right)

target_img = img_left if apply_to == "left" else img_right
h, w = target_img.shape

# --- pick wound center ---
ys, xs = np.where(~np.isnan(target_img))  # valid foot pixels only
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

# --- pick wound shape mode randomly ---
shape_modes = ["circle", "multi"]
shape_mode = np.random.choice(shape_modes)

print(f"Generated wound at ({y_center},{x_center}) with shape {shape_mode}")

Y, X = np.ogrid[:h, :w]
mask_core_binary   = np.zeros((h, w), dtype=bool)
mask_inflam_binary = np.zeros((h, w), dtype=bool)

if shape_mode == "circle":
    mask_core_binary = (X - x_center)**2 + (Y - y_center)**2 <= core_radii[0]**2
    mask_inflam_binary = (X - x_center)**2 + (Y - y_center)**2 <= inflam_radii[0]**2

elif shape_mode == "multi":
    n_blobs = np.random.randint(2, 11)  # 2–10 blobs
    print(f"Generated {n_blobs} blobs for multi shape")

    # Store centers of placed blobs
    blob_centers = [(x_center, y_center)]

    # Place initial blob
    mask_core_binary |= (X - x_center)**2 + (Y - y_center)**2 <= core_radii[0]**2
    mask_inflam_binary |= (X - x_center)**2 + (Y - y_center)**2 <= inflam_radii[0]**2

    for _ in range(n_blobs - 1):
        # Pick an existing blob center to attach to
        base_x, base_y = blob_centers[np.random.randint(len(blob_centers))]

        # Pick random angle + distance (to ensure touching but not always full overlap)
        angle = np.random.uniform(0, 2*np.pi)
        dist = np.random.randint(int(core_radii[0]*0.8), int(core_radii[0]*1.5))

        dx = int(np.cos(angle) * dist)
        dy = int(np.sin(angle) * dist)

        new_x = base_x + dx
        new_y = base_y + dy
        blob_centers.append((new_x, new_y))

        # Randomize radii a bit for irregularity
        r_core   = np.random.randint(core_radii[0]//2, core_radii[0])
        r_inflam = np.random.randint(inflam_radii[0]//2, inflam_radii[0])

        # Build new blob
        blob_core = (X - new_x)**2 + (Y - new_y)**2 <= r_core**2
        blob_inflam = (X - new_x)**2 + (Y - new_y)**2 <= r_inflam**2

        # Merge with cluster
        mask_core_binary   |= blob_core
        mask_inflam_binary |= blob_inflam

# --- convert binary masks to temperature masks ---
mask_core   = np.zeros((h, w))
mask_inflam = np.zeros((h, w))

mask_core[mask_core_binary]     = core_temp
mask_inflam[mask_inflam_binary] = inflam_temp

# --- apply Gaussian blur ---
mask_core   = gaussian_filter(mask_core, sigma=blur_sigma_core)
mask_inflam = gaussian_filter(mask_inflam, sigma=blur_sigma_inflam)

# --- final mask ---
mask = np.maximum(mask_inflam, mask_core)

# Save once
scipy.io.savemat(os.path.join(output_dir, "wound_mask_locked.mat"),
                 {"wound_mask": mask})

# ==========================
# Loop through days and apply SAME wound
# ==========================

for i in range(num_days):
    scan_left  = left_crop[i, 0]
    scan_right = right_crop[i, 0]

    img_left  = np.where(scan_left == 0, np.nan, scan_left)
    img_right = np.where(scan_right == 0, np.nan, scan_right)

    if apply_to == "left":
        img_left_with_wound  = np.where(~np.isnan(img_left), img_left + mask, np.nan)
        img_right_with_wound = img_right
    else:
        img_left_with_wound  = img_left
        img_right_with_wound = np.where(~np.isnan(img_right), img_right + mask, np.nan)

    # Trim & combine
    img_left_trim  = trim_empty_columns(img_left_with_wound)
    img_right_trim = trim_empty_columns(img_right_with_wound)

    gap = np.full((img_left_trim.shape[0], 5), np.nan)
    combined = np.hstack((img_left_trim, gap, img_right_trim))

    # Plot and save
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(combined, cmap="hot")
    ax.axis("off")
    ax.set_title(f"Day {i+1}: wound on {apply_to}")

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
    cbar.set_label("Temperature (°C)")

    plt.savefig(os.path.join(output_dir, f"foot_day{i+1}_with_wound.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

print(f"Saved {num_days} PNGs in '{output_dir}', all with the same wound mask on {apply_to} foot")
