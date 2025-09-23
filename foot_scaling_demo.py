# foot_scaling_demo.py
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.io.matlab import MatReadWarning
from scipy.ndimage import zoom

# ==========================
# CONFIGURATION
# ==========================
MAT_FILE = "Data/Temp Data/pnt_mat_files/pnt1.mat"
OUTPUT_DIR = "output_scaling_demo"
CMAP = "hot"

# ==========================
# SCALING FUNCTIONS
# ==========================
def scale_image_preserve_temps(img, scale_x, scale_y):
    """Scale image while preserving temperature values"""
    # Create mask of valid pixels
    valid_mask = ~np.isnan(img)
    
    # Replace NaN with a safe value for scaling
    min_temp = np.nanmin(img)
    if np.isnan(min_temp):
        fill_value = 20.0
    else:
        fill_value = min_temp - 5.0
    
    img_filled = np.where(valid_mask, img, fill_value)
    
    # Scale the image and mask separately
    scaled_img = zoom(img_filled, (scale_y, scale_x), order=1, mode='nearest')
    scaled_mask = zoom(valid_mask.astype(float), (scale_y, scale_x), order=1, mode='nearest')
    
    # Restore NaN values
    final_img = np.where(scaled_mask > 0.5, scaled_img, np.nan)
    
    return final_img

def calculate_scaling_factors(small_img, large_img):
    """Calculate scaling factors"""
    large_h, large_w = large_img.shape
    small_h, small_w = small_img.shape
    
    scale_x = large_w / small_w
    scale_y = large_h / small_h
    
    return scale_x, scale_y

def transform_coordinate_to_original(scaled_x, scaled_y, scale_x, scale_y):
    """Convert scaled coordinate back to original"""
    return scaled_x / scale_x, scaled_y / scale_y

def transform_coordinate_to_scaled(x, y, scale_x, scale_y):
    """Convert original coordinate to scaled"""
    return x * scale_x, y * scale_y

# ==========================
# HELPER FUNCTIONS
# ==========================
def to_nan(img):
    return np.where(img == 0, np.nan, img)

def trim_to_content(img):
    if img is None or img.size == 0:
        return img
    
    if img.ndim == 0:
        return img
    elif img.ndim == 1:
        img = img.reshape(-1, 1)
    
    valid_rows = ~(np.all(np.isnan(img), axis=1))
    valid_cols = ~(np.all(np.isnan(img), axis=0))
    if not np.any(valid_rows) or not np.any(valid_cols):
        return img
    return img[valid_rows][:, valid_cols]

def mirror_horiz(img):
    if img is None or img.ndim < 2:
        return img
    return np.fliplr(img)

# ==========================
# DEMO FUNCTION
# ==========================
def demonstrate_scaling():
    warnings.filterwarnings("ignore", category=MatReadWarning)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    mat = scipy.io.loadmat(MAT_FILE)
    left_crop = mat["Indirect_plantar_Right_crop"]
    right_crop = mat["Indirect_plantar_Left_crop"]
    
    # Process images
    img_left = trim_to_content(to_nan(left_crop))
    img_right = trim_to_content(to_nan(right_crop))
    img_right_mir = mirror_horiz(img_right)
    
    print(f"Original left foot shape: {img_left.shape}")
    print(f"Original right foot (mirrored) shape: {img_right_mir.shape}")
    
    # Determine which is smaller
    left_size = img_left.shape[0] * img_left.shape[1]
    right_size = img_right_mir.shape[0] * img_right_mir.shape[1]
    
    if left_size < right_size:
        smaller_img = img_left
        larger_img = img_right_mir
        smaller_name = "Left Foot"
        larger_name = "Right Foot (mirrored)"
    else:
        smaller_img = img_right_mir
        larger_img = img_left
        smaller_name = "Right Foot (mirrored)"
        larger_name = "Left Foot"
    
    print(f"\\n{smaller_name} is smaller ({smaller_img.shape}) - scaling to match {larger_name} ({larger_img.shape})")
    
    # Calculate scaling factors
    scale_x, scale_y = calculate_scaling_factors(smaller_img, larger_img)
    print(f"Scaling factors: x={scale_x:.3f}, y={scale_y:.3f}")
    
    # Scale the smaller image
    scaled_smaller = scale_image_preserve_temps(smaller_img, scale_x, scale_y)
    print(f"Scaled {smaller_name} shape: {scaled_smaller.shape}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Row 1: Original images
    im1 = axes[0, 0].imshow(img_left, cmap=CMAP)
    axes[0, 0].set_title('Original Left Foot')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0])
    
    im2 = axes[0, 1].imshow(img_right_mir, cmap=CMAP)
    axes[0, 1].set_title('Original Right Foot (mirrored)')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Show size difference
    axes[0, 2].text(0.1, 0.8, f"Size Comparison:", fontsize=14, fontweight='bold')
    axes[0, 2].text(0.1, 0.6, f"Left: {img_left.shape} = {left_size} pixels", fontsize=12)
    axes[0, 2].text(0.1, 0.4, f"Right: {img_right_mir.shape} = {right_size} pixels", fontsize=12)
    axes[0, 2].text(0.1, 0.2, f"Scale factors: x={scale_x:.3f}, y={scale_y:.3f}", fontsize=12)
    axes[0, 2].axis('off')
    
    # Row 2: Scaled comparison
    if left_size < right_size:
        im3 = axes[1, 0].imshow(scaled_smaller, cmap=CMAP)
        axes[1, 0].set_title('Left Foot (scaled up)')
        im4 = axes[1, 1].imshow(larger_img, cmap=CMAP)
        axes[1, 1].set_title('Right Foot (original)')
    else:
        im3 = axes[1, 0].imshow(larger_img, cmap=CMAP)
        axes[1, 0].set_title('Left Foot (original)')
        im4 = axes[1, 1].imshow(scaled_smaller, cmap=CMAP)
        axes[1, 1].set_title('Right Foot (scaled up)')
    
    axes[1, 0].axis('off')
    axes[1, 1].axis('off')
    plt.colorbar(im3, ax=axes[1, 0])
    plt.colorbar(im4, ax=axes[1, 1])
    
    # Demonstrate coordinate transformation
    # Pick a sample point on the smaller (original) image
    sample_y, sample_x = np.unravel_index(np.nanargmax(smaller_img), smaller_img.shape)
    
    # Transform to scaled coordinates
    scaled_sample_x, scaled_sample_y = transform_coordinate_to_scaled(
        sample_x, sample_y, scale_x, scale_y
    )
    
    # Transform back to original
    back_to_orig_x, back_to_orig_y = transform_coordinate_to_original(
        scaled_sample_x, scaled_sample_y, scale_x, scale_y
    )
    
    axes[1, 2].text(0.1, 0.8, f"Coordinate Transformation Demo:", fontsize=12, fontweight='bold')
    axes[1, 2].text(0.1, 0.6, f"Original point: ({sample_x:.1f}, {sample_y:.1f})", fontsize=10)
    axes[1, 2].text(0.1, 0.5, f"Scaled point: ({scaled_sample_x:.1f}, {scaled_sample_y:.1f})", fontsize=10)
    axes[1, 2].text(0.1, 0.4, f"Back to original: ({back_to_orig_x:.1f}, {back_to_orig_y:.1f})", fontsize=10)
    axes[1, 2].text(0.1, 0.2, f"Transform accuracy: ±{abs(back_to_orig_x-sample_x):.3f}px", fontsize=10)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "scaling_demonstration.png"), dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\\n=== COORDINATE TRANSFORMATION TEST ===")
    print(f"Original coordinates: ({sample_x:.1f}, {sample_y:.1f})")
    print(f"Scaled coordinates: ({scaled_sample_x:.1f}, {scaled_sample_y:.1f})")
    print(f"Back to original: ({back_to_orig_x:.1f}, {back_to_orig_y:.1f})")
    print(f"Transformation accuracy: ±{max(abs(back_to_orig_x-sample_x), abs(back_to_orig_y-sample_y)):.6f} pixels")
    
    return {
        'smaller_img': smaller_img,
        'larger_img': larger_img,
        'scaled_smaller': scaled_smaller,
        'scale_x': scale_x,
        'scale_y': scale_y,
        'smaller_name': smaller_name,
        'larger_name': larger_name
    }

if __name__ == "__main__":
    results = demonstrate_scaling()