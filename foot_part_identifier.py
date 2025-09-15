import scipy.io
import os
import numpy as np
import matplotlib.pyplot as plt

from generate_heatspot_mask import trim_empty_columns

# ===========================
# Fixed Variables for Testing
# ===========================
day_index = 0
pnt = "Patient 1"
output_dir = "foot_images_folder"

# ===========================
# Helper Functions
# ===========================

# ---------------------------

# Directories #
## MAT-Files ##
mat_folder_path = "./Data/Temp Data"
if __name__ == "__main__":

    ## Testing With Healthy 1 ##
    mat_pnt_one = scipy.io.loadmat(mat_folder_path + "/gz1.mat")

    # Working on right foot for now #
    right_crop = mat_pnt_one["Direct_plantar_Right_crop"]
    scan_right = right_crop[day_index, 0]
    img_right = np.where(scan_right == 0, np.nan, scan_right)


    # ==========================
    # Plot and save
    # ==========================

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(img_right, cmap="hot")
    ax.axis("off")
    ax.set_title(f"Day {day_index+1}: Foot for {pnt}")

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
    cbar.set_label("Temperature (°C)")

    plt.savefig(os.path.join(output_dir, f"foot_day{day_index+1}png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    os.makedirs(output_dir, exist_ok=True)
    scipy.io.savemat(os.path.join(output_dir, f"{pnt}{day_index+1}.mat"),
                    {"foot_image": img_right})

    print(f"Saved combined PNG in '{output_dir}' ({pnt})")