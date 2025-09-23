# foot_overlay_creator.py
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
MAT_FILE = "Data/Temp Data/gz1.mat"
OUTPUT_DIR = "output_overlay_system"
CMAP = "hot"
RIGHT_ALPHA = 0.45  # Transparency for right foot overlay

# ==========================
# FUNCTIONS
# ==========================
def scale_image_preserve_temps(img, scale_x, scale_y):
    """Scale image while preserving temperature values"""
    valid_mask = ~np.isnan(img)
    min_temp = np.nanmin(img)
    fill_value = min_temp - 5.0 if not np.isnan(min_temp) else 20.0
    
    img_filled = np.where(valid_mask, img, fill_value)
    scaled_img = zoom(img_filled, (scale_y, scale_x), order=1, mode='nearest')
    scaled_mask = zoom(valid_mask.astype(float), (scale_y, scale_x), order=1, mode='nearest')
    
    return np.where(scaled_mask > 0.5, scaled_img, np.nan)

def to_nan(img):
    """Convert 0 to NaN, handling different data structures"""
    if img is None:
        return img
    
    # Handle scalar values
    if np.isscalar(img):
        return np.nan if img == 0 else img
    
    # Handle arrays
    img = np.array(img)
    if img.size == 0:
        return img
    
    # For object arrays or nested structures, try to extract the actual image
    if img.dtype == object:
        # This is likely an array of MATLAB objects
        return img
    
    # For regular numeric arrays
    return np.where(img == 0, np.nan, img)

def trim_to_content(img):
    if img is None or img.size == 0:
        return img
    if img.ndim < 2:
        return img
    
    valid_rows = ~np.all(np.isnan(img), axis=1)
    valid_cols = ~np.all(np.isnan(img), axis=0)
    return img[valid_rows][:, valid_cols] if np.any(valid_rows) and np.any(valid_cols) else img

def mirror_horiz(img):
    return np.fliplr(img) if img is not None and img.ndim >= 2 else img

def pad_to_same_size(img1, img2):
    """Pad both images to the same size (centered)"""
    H = max(img1.shape[0], img2.shape[0])
    W = max(img1.shape[1], img2.shape[1])
    
    def center_pad(img, target_h, target_w):
        canvas = np.full((target_h, target_w), np.nan, dtype=float)
        h, w = img.shape
        top = max(0, (target_h - h) // 2)
        left = max(0, (target_w - w) // 2)
        canvas[top:top+h, left:left+w] = img
        return canvas
    
    return center_pad(img1, H, W), center_pad(img2, H, W)

def create_foot_overlay():
    """Create overlay of original left foot with scaled mirrored right foot"""
    warnings.filterwarnings("ignore", category=MatReadWarning)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    mat = scipy.io.loadmat(MAT_FILE)
    print("Available keys:", list(mat.keys()))
    
    left_crop = mat["Indirect_plantar_Right_crop"]  # Note: naming convention
    right_crop = mat["Indirect_plantar_Left_crop"]
    
    print(f"Raw data shapes - Left: {left_crop.shape}, Right: {right_crop.shape}")
    print(f"Raw data types - Left: {left_crop.dtype}, Right: {right_crop.dtype}")
    
    # Handle different data structures
    if left_crop.shape == (10, 1) and left_crop.dtype == object:
        # This appears to be an array of objects containing the actual images
        print("Detected object array structure - extracting first valid image")
        
        # Try to extract the first valid image from each array
        scan_left = None
        scan_right = None
        
        for i in range(left_crop.shape[0]):
            try:
                candidate_left = left_crop[i, 0]
                candidate_right = right_crop[i, 0]
                
                # Check if these are valid 2D arrays
                if (hasattr(candidate_left, 'shape') and len(candidate_left.shape) == 2 and 
                    hasattr(candidate_right, 'shape') and len(candidate_right.shape) == 2):
                    scan_left = candidate_left
                    scan_right = candidate_right
                    print(f"Using images from index {i}")
                    print(f"Left shape: {scan_left.shape}, Right shape: {scan_right.shape}")
                    break
            except:
                continue
        
        if scan_left is None or scan_right is None:
            print("Error: Could not extract valid images from object arrays")
            return None
    else:
        # Direct 2D image data
        scan_left = left_crop
        scan_right = right_crop
    
    # Process images
    img_left = trim_to_content(to_nan(scan_left))
    img_right = trim_to_content(to_nan(scan_right))
    img_right_mir = mirror_horiz(img_right)  # Mirror the right foot
    
    print(f"Processed shapes - Left: {img_left.shape}, Right mirrored: {img_right_mir.shape}")
    
    # Determine scaling needed
    left_size = img_left.shape[0] * img_left.shape[1]
    right_size = img_right_mir.shape[0] * img_right_mir.shape[1]
    
    print(f"Pixel counts - Left: {left_size}, Right: {right_size}")
    
    # Always scale right foot to match left foot for overlay
    if img_right_mir.shape != img_left.shape:
        scale_x = img_left.shape[1] / img_right_mir.shape[1]
        scale_y = img_left.shape[0] / img_right_mir.shape[0]
        
        print(f"Scaling right foot - factors: x={scale_x:.3f}, y={scale_y:.3f}")
        
        img_right_scaled = scale_image_preserve_temps(img_right_mir, scale_x, scale_y)
        print(f"Right foot scaled to: {img_right_scaled.shape}")
    else:
        img_right_scaled = img_right_mir
        scale_x = scale_y = 1.0
        print("No scaling needed - feet are same size")
    
    # Pad both to same canvas size for perfect alignment
    left_canvas, right_canvas = pad_to_same_size(img_left, img_right_scaled)
    
    print(f"Final canvas size: {left_canvas.shape}")
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Row 1: Individual feet
    im1 = axes[0, 0].imshow(img_left, cmap=CMAP)
    axes[0, 0].set_title('Original Left Foot')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0])
    
    im2 = axes[0, 1].imshow(img_right_mir, cmap=CMAP)
    axes[0, 1].set_title('Original Right Foot (Mirrored)')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1])
    
    im3 = axes[0, 2].imshow(img_right_scaled, cmap=CMAP)
    axes[0, 2].set_title('Right Foot (Scaled & Mirrored)')
    axes[0, 2].axis('off')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # Row 2: Overlays with different transparencies
    alphas = [0.3, 0.45, 0.6]
    titles = ['Light Overlay (α=0.3)', 'Medium Overlay (α=0.45)', 'Heavy Overlay (α=0.6)']
    
    for i, (alpha, title) in enumerate(zip(alphas, titles)):
        axes[1, i].imshow(left_canvas, cmap=CMAP)  # Base: left foot
        axes[1, i].imshow(right_canvas, cmap=CMAP, alpha=alpha)  # Overlay: right foot
        axes[1, i].set_title(title)
        axes[1, i].axis('off')
        
        # Add colorbar to middle overlay
        if i == 1:
            im_overlay = axes[1, i].imshow(left_canvas, cmap=CMAP)
            plt.colorbar(im_overlay, ax=axes[1, i])
    
    plt.tight_layout()
    
    # Save the visualization
    output_path = os.path.join(OUTPUT_DIR, "foot_overlay_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\\nOverlay visualization saved to: {output_path}")
    
    plt.show()
    
    # Create a focused overlay for analysis
    fig2, ax = plt.subplots(figsize=(10, 8))
    
    # Main overlay
    im_base = ax.imshow(left_canvas, cmap=CMAP)
    im_overlay = ax.imshow(right_canvas, cmap=CMAP, alpha=RIGHT_ALPHA)
    
    ax.set_title(f'Foot Overlay Analysis\\nLeft Foot (original) + Right Foot (scaled α={RIGHT_ALPHA})\\n'
                f'Scale factors: x={scale_x:.3f}, y={scale_y:.3f}')
    ax.axis('off')
    
    # Add colorbar
    cbar = plt.colorbar(im_base, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Temperature (°C)')
    
    # Save focused overlay
    focused_path = os.path.join(OUTPUT_DIR, "focused_foot_overlay.png")
    plt.savefig(focused_path, dpi=300, bbox_inches='tight')
    print(f"Focused overlay saved to: {focused_path}")
    
    plt.show()
    
    print("\\n" + "="*60)
    print("OVERLAY SUMMARY")
    print("="*60)
    print(f"Original left foot size: {img_left.shape}")
    print(f"Original right foot size: {img_right_mir.shape}")
    print(f"Scaled right foot size: {img_right_scaled.shape}")
    print(f"Final canvas size: {left_canvas.shape}")
    print(f"Scaling factors applied: x={scale_x:.3f}, y={scale_y:.3f}")
    print(f"Right foot transparency: α={RIGHT_ALPHA}")
    
    return {
        'left_original': img_left,
        'right_mirrored': img_right_mir,
        'right_scaled': img_right_scaled,
        'left_canvas': left_canvas,
        'right_canvas': right_canvas,
        'scale_x': scale_x,
        'scale_y': scale_y
    }

if __name__ == "__main__":
    results = create_foot_overlay()