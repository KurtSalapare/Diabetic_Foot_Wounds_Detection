import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import os

# Load the .mat file
mat = scipy.io.loadmat("Data/Temp Data/gz1.mat")

# Access the right plantar crop data (10x1 cell array)
right_crop = mat["Direct_plantar_Right_crop"]

# Create output folder
output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)

# Loop through all days (cells)
num_days = right_crop.shape[0]
for i in range(num_days):
    # Extract one day's scan (288x382 double)
    scan = right_crop[i, 0]

    # Replace zeros (cropped background) with NaN for visualization
    img = np.where(scan == 0, np.nan, scan)

    # Plot and save as PNG
    plt.imshow(img, cmap="hot")
    plt.colorbar(label="Temperature (°C)")
    plt.title(f"Foot scan - Day {i+1}")
    plt.savefig(os.path.join(output_dir, f"foot_day{i+1}.png"), dpi=300)
    plt.close()  # Close figure to avoid memory issues

print(f"Saved {num_days} images in '{output_dir}'")
