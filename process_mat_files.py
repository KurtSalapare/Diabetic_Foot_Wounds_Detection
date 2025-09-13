import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
from scipy.io.matlab.miobase import MatReadWarning

# Silence the MAT file warning
warnings.filterwarnings("ignore", category=MatReadWarning)

# Load the .mat file
mat = scipy.io.loadmat("Data/Temp Data/gz10.mat")

# NOTE: swapped left/right as you mentioned
left_crop  = mat["Direct_plantar_Right_crop"]
right_crop = mat["Direct_plantar_Left_crop"]

# Create output folder
output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)

def trim_empty_columns(img):
    """Remove fully-zero/NaN columns to bring feet closer."""
    valid_cols = ~(np.all(np.isnan(img) | (img == 0), axis=0))
    return img[:, valid_cols]

num_days = left_crop.shape[0]

for i in range(num_days):
    scan_left  = left_crop[i, 0]
    scan_right = right_crop[i, 0]

    img_left  = np.where(scan_left == 0, np.nan, scan_left)
    img_right = np.where(scan_right == 0, np.nan, scan_right)

    # Trim empty columns so feet don't look so far apart
    img_left = trim_empty_columns(img_left)
    img_right = trim_empty_columns(img_right)

    # Combine into one array, add small gap (5 columns of NaN)
    gap = np.full((img_left.shape[0], 5), np.nan)
    combined = np.hstack((img_left, gap, img_right))

    # Plot with larger figure and thinner colorbar
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(combined, cmap="hot")
    ax.axis("off")
    ax.set_title(f"Foot scans - Day {i+1}")

    # Add a thinner colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
    cbar.set_label("Temperature (°C)")

    plt.savefig(os.path.join(output_dir, f"foot_day{i+1}_combined.png"), dpi=300, bbox_inches="tight")
    plt.close()

print(f"Saved {num_days} combined images in '{output_dir}'")
