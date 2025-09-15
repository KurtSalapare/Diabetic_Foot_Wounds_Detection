import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import os

# Load both original and modified files for consistent scaling
mat_original = scipy.io.loadmat("Data/Temp Data/gz1.mat")
mat_modified = scipy.io.loadmat("Data/Temp Data/Modified/gz1_modified.mat")

# Access the data from both files
right_crop_original = mat_original["Direct_plantar_Left_crop"]
right_crop_modified = mat_modified["Direct_plantar_Left_crop"]

# Find global min and max temperatures across ALL frames for consistent scaling
all_temps_original = []
all_temps_modified = []

for i in range(right_crop_original.shape[0]):
    scan_original = right_crop_original[i, 0]
    scan_modified = right_crop_modified[i, 0]
    
    # Only consider non-zero pixels (foot region)
    all_temps_original.extend(scan_original[scan_original != 0])
    all_temps_modified.extend(scan_modified[scan_modified != 0])

# Calculate global temperature range
global_min = min(np.min(all_temps_original), np.min(all_temps_modified))
global_max = max(np.max(all_temps_original), np.max(all_temps_modified))

print(f"Global temperature range: {global_min:.2f} to {global_max:.2f} °C")

# Create output folders
output_dir_original = "output_images_direct_plantar_left_original"
output_dir_modified = "output_images_direct_plantar_left_modified"
os.makedirs(output_dir_original, exist_ok=True)
os.makedirs(output_dir_modified, exist_ok=True)

# Function to plot with consistent scaling
def plot_scan(scan, output_path, title, vmin, vmax):
    img = np.where(scan == 0, np.nan, scan)
    plt.imshow(img, cmap="hot", vmin=vmin, vmax=vmax)
    plt.colorbar(label="Temperature (°C)")
    plt.title(title)
    plt.savefig(output_path, dpi=300)
    plt.close()

# Loop through all days and plot both original and modified with consistent scaling
num_days = right_crop_original.shape[0]
for i in range(num_days):
    scan_original = right_crop_original[i, 0]
    scan_modified = right_crop_modified[i, 0]
    
    # Plot original with consistent scaling
    plot_scan(scan_original, 
              os.path.join(output_dir_original, f"foot_day{i+1}.png"),
              f"Original Foot scan - Day {i+1}",
              global_min, global_max)
    
    # Plot modified with consistent scaling
    plot_scan(scan_modified, 
              os.path.join(output_dir_modified, f"foot_day{i+1}_modified.png"),
              f"Modified Foot scan - Day {i+1}",
              global_min, global_max)

print(f"Saved {num_days} original images in '{output_dir_original}'")
print(f"Saved {num_days} modified images in '{output_dir_modified}'")