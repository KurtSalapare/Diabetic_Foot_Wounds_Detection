
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.io.matlab import MatReadWarning
from scipy.ndimage import zoom, rotate

# ==========================
# CONFIGURATION
# ==========================
# You can easily switch between different MAT files here
AVAILABLE_MAT_FILES = {
    "gz1": "Data/Temp Data/gz1.mat",
    "gz2": "Data/Temp Data/gz2.mat",
    "gz7": "Data/Temp Data/gz7.mat",
    "gz8": "Data/Temp Data/gz8.mat", 
    "gz9": "Data/Temp Data/gz9.mat",
    "pnt1": "Data/Temp Data/pnt_mat_files/pnt1.mat",
    "pnt2": "Data/Temp Data/pnt_mat_files/pnt2.mat",
    "pnt3": "Data/Temp Data/pnt_mat_files/pnt3.mat",
    # Add more files as needed - the system will auto-detect their structure
}

# Select which file to use (change this key to switch files)
SELECTED_FILE = "gz1"  # Change to "gz1", "pnt1", etc.

# For object arrays, you can manually specify which index to use (set to None for automatic selection)
MANUAL_INDEX_SELECTION = None  # Set to 0, 1, 2, etc. to manually select from object array

# Rotation optimization settings
ENABLE_ROTATION_OPTIMIZATION = True  # Set to False to skip rotation
ROTATION_ANGLE_RANGE = (-30, 30)  # Range of angles to test (degrees)
ROTATION_ANGLE_STEP = 1  # Step size for angle testing (degrees) - 1° for finer precision

MAT_FILE = AVAILABLE_MAT_FILES[SELECTED_FILE]
OUTPUT_DIR = "output_overlay_system"  # Single output folder for all files
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

def to_nan(img, adaptive_threshold=True):
    """
    Convert temperature values to NaN where appropriate.
    Handles different data structures including object arrays robustly.
    """
    if img is None:
        return img
    
    # Handle scalar values
    if np.isscalar(img):
        return np.nan if img < 25 else img  # Use temperature threshold
    
    # Handle object arrays by extracting numeric data first
    if hasattr(img, 'dtype') and img.dtype == object:
        try:
            # Try to convert object array to numeric
            img = np.array(img, dtype=float)
        except (ValueError, TypeError):
            # If conversion fails, the data might already be processed
            print("Warning: Could not convert object array to numeric, returning as-is")
            return img
    
    # Convert to array if not already
    if not isinstance(img, np.ndarray):
        try:
            img = np.array(img, dtype=float)
        except:
            print("Warning: Could not convert to numeric array")
            return img
    
    if img.size == 0:
        return img
    
    # For numeric arrays, apply adaptive or fixed threshold
    try:
        if adaptive_threshold:
            # Use adaptive threshold based on data distribution
            valid_data = img[~np.isnan(img)]
            if len(valid_data) > 0:
                min_temp = np.min(valid_data)
                max_temp = np.max(valid_data)
                temp_range = max_temp - min_temp
                
                # If temperature range suggests thermal data (reasonable body temp range)
                if temp_range > 5 and min_temp > 15 and max_temp < 50:
                    # Use 5% above minimum as threshold for thermal data
                    threshold = min_temp + 0.05 * temp_range
                    print(f"Adaptive threshold: {threshold:.1f}°C (range: {min_temp:.1f}-{max_temp:.1f}°C)")
                    return np.where(img < threshold, np.nan, img)
                else:
                    # Use zero threshold for non-thermal data
                    return np.where(img == 0, np.nan, img)
            else:
                return img
        else:
            # Use fixed 25°C threshold
            return np.where(img < 25, np.nan, img)
    except:
        # Fallback to zero threshold if all else fails
        return np.where(img == 0, np.nan, img)

def trim_to_content(img):
    """Trim rows/cols that are entirely NaN, with robust error handling"""
    if img is None:
        return img
    
    # Convert to array if needed
    if not isinstance(img, np.ndarray):
        try:
            img = np.array(img, dtype=float)
        except:
            print("Warning: Could not convert to array in trim_to_content")
            return img
    
    if img.size == 0:
        return img
    
    if img.ndim < 2:
        return img
    
    try:
        # Check for valid rows and columns
        valid_rows = ~np.all(np.isnan(img), axis=1)
        valid_cols = ~np.all(np.isnan(img), axis=0)
        
        # Return trimmed image if we have valid regions
        if np.any(valid_rows) and np.any(valid_cols):
            return img[valid_rows][:, valid_cols]
        else:
            return img
    except Exception as e:
        print(f"Warning: Error in trim_to_content: {str(e)}")
        return img

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

def rotate_image_preserve_temps(img, angle):
    """Rotate image while preserving temperature values"""
    if angle == 0:
        return img
    
    # Handle NaN values by filling temporarily during rotation
    valid_mask = ~np.isnan(img)
    min_temp = np.nanmin(img)
    fill_value = min_temp - 10.0 if not np.isnan(min_temp) else 15.0
    
    # Fill NaN values temporarily
    img_filled = np.where(valid_mask, img, fill_value)
    
    # Rotate the image and mask
    rotated_img = rotate(img_filled, angle, order=1, mode='constant', cval=fill_value, reshape=True)
    rotated_mask = rotate(valid_mask.astype(float), angle, order=1, mode='constant', cval=0, reshape=True)
    
    # Restore NaN values where the mask indicates invalid data
    return np.where(rotated_mask > 0.5, rotated_img, np.nan)

def create_binary_mask(img, adaptive_threshold=True):
    """Create binary mask from thermal image"""
    if img is None:
        return None
    
    # Get valid (non-NaN) data
    valid_data = img[~np.isnan(img)]
    if len(valid_data) == 0:
        return np.zeros_like(img, dtype=bool)
    
    if adaptive_threshold:
        # Use adaptive threshold similar to to_nan function
        min_temp = np.min(valid_data)
        max_temp = np.max(valid_data)
        temp_range = max_temp - min_temp
        
        if temp_range > 5 and min_temp > 15 and max_temp < 50:
            # Use 5% above minimum as threshold for thermal data
            threshold = min_temp + 0.05 * temp_range
        else:
            # Use zero threshold for non-thermal data
            threshold = 0
    else:
        threshold = 25  # Fixed threshold
    
    # Create binary mask: True where thermal data exists above threshold
    mask = (~np.isnan(img)) & (img > threshold)
    return mask

def calculate_overlap_score(mask1, mask2):
    """Calculate Jaccard index (IoU) between two binary masks"""
    if mask1.shape != mask2.shape:
        return 0
    
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    if union == 0:
        return 0
    
    return intersection / union

def find_best_rotation_angle(left_foot, right_foot_mirrored, angle_range=(-30, 30), angle_step=2):
    """
    Find the rotation angle that maximizes overlap between feet using binary masks
    
    Parameters:
    - left_foot: Reference foot (stays fixed)
    - right_foot_mirrored: Foot to rotate
    - angle_range: Range of angles to test (degrees)
    - angle_step: Step size for angle testing
    
    Returns:
    - best_angle: Optimal rotation angle
    - best_score: Maximum overlap score achieved
    - scores: Dictionary of all tested angles and their scores
    """
    print(f"Testing rotation angles from {angle_range[0]}° to {angle_range[1]}° (step: {angle_step}°)")
    
    # Create binary mask for the reference left foot
    left_mask = create_binary_mask(left_foot, adaptive_threshold=True)
    
    best_angle = 0
    best_score = 0
    scores = {}
    
    # Test different rotation angles
    for angle in range(angle_range[0], angle_range[1] + 1, angle_step):
        # Rotate the right foot
        rotated_right = rotate_image_preserve_temps(right_foot_mirrored, angle)
        
        # Create binary mask for rotated right foot
        right_mask = create_binary_mask(rotated_right, adaptive_threshold=True)
        
        # Pad both masks to same size for fair comparison
        if left_mask.shape != right_mask.shape:
            padded_left, padded_right = pad_to_same_size(left_mask.astype(float), right_mask.astype(float))
            left_mask_padded = padded_left > 0.5
            right_mask_padded = padded_right > 0.5
        else:
            left_mask_padded = left_mask
            right_mask_padded = right_mask
        
        # Calculate overlap score
        score = calculate_overlap_score(left_mask_padded, right_mask_padded)
        scores[angle] = score
        
        if score > best_score:
            best_score = score
            best_angle = angle
    
    return best_angle, best_score, scores

def detect_and_extract_images(left_crop, right_crop):
    """
    Automatically detect and extract valid 2D images from various MAT file structures.
    Handles different object array sizes and nested structures.
    """
    print(f"Raw data shapes - Left: {left_crop.shape}, Right: {right_crop.shape}")
    print(f"Raw data types - Left: {left_crop.dtype}, Right: {right_crop.dtype}")
    
    # Case 1: Direct 2D numeric arrays
    if (left_crop.dtype != object and len(left_crop.shape) == 2 and 
        left_crop.shape[0] > 50 and left_crop.shape[1] > 50):
        print("Detected direct 2D numeric arrays")
        return left_crop, right_crop
    
    # Case 2: Object arrays (various sizes: (8,1), (10,1), etc.)
    if (left_crop.dtype == object and len(left_crop.shape) == 2 and left_crop.shape[1] == 1):
        print(f"Detected object array structure ({left_crop.shape[0]}, 1) - extracting valid images")
        
        # Debug: Show what's in each index to help identify the best data
        print("Analyzing object array contents:")
        candidates = []
        for i in range(left_crop.shape[0]):
            try:
                candidate_left = left_crop[i, 0]
                candidate_right = right_crop[i, 0]
                
                if (hasattr(candidate_left, 'shape') and len(candidate_left.shape) == 2 and 
                    hasattr(candidate_right, 'shape') and len(candidate_right.shape) == 2):
                    
                    left_shape = candidate_left.shape
                    right_shape = candidate_right.shape
                    left_size = left_shape[0] * left_shape[1]
                    right_size = right_shape[0] * right_shape[1]
                    
                    # Check temperature ranges to identify thermal data
                    left_temp_range = f"{np.nanmin(candidate_left):.1f}-{np.nanmax(candidate_left):.1f}"
                    right_temp_range = f"{np.nanmin(candidate_right):.1f}-{np.nanmax(candidate_right):.1f}"
                    
                    print(f"  Index {i}: Left {left_shape} ({left_size} pixels, temps {left_temp_range}), Right {right_shape} ({right_size} pixels, temps {right_temp_range})")
                    
                    # Store candidate with quality metrics
                    if left_shape[0] > 50 and left_shape[1] > 50 and right_shape[0] > 50 and right_shape[1] > 50:
                        candidates.append((i, candidate_left, candidate_right, left_size + right_size))
                        
            except Exception as e:
                print(f"  Index {i}: Error - {str(e)}")
                continue
        
        # Check for manual index selection first
        if MANUAL_INDEX_SELECTION is not None and 0 <= MANUAL_INDEX_SELECTION < len(candidates):
            manual_idx, manual_left, manual_right, manual_pixels = candidates[MANUAL_INDEX_SELECTION]
            print(f"Using manual selection: index {MANUAL_INDEX_SELECTION}")
            print(f"Left shape: {manual_left.shape}, Right shape: {manual_right.shape}")
            return manual_left, manual_right
        
        # Select the best candidate based on thermal characteristics
        if candidates:
            # If all have same pixel count, select based on temperature range and quality
            best_candidate = None
            best_score = -1
            
            for i, candidate_left, candidate_right, total_pixels in candidates:
                # Calculate quality score based on temperature characteristics
                left_temps = candidate_left[candidate_left > 0]  # Non-zero temperatures
                right_temps = candidate_right[candidate_right > 0]
                
                if len(left_temps) > 0 and len(right_temps) > 0:
                    # Score based on temperature range (prefer realistic body temp ranges)
                    left_range = np.max(left_temps) - np.min(left_temps)
                    right_range = np.max(right_temps) - np.min(right_temps)
                    avg_range = (left_range + right_range) / 2
                    
                    # Score based on data density (less zeros/NaNs is better)
                    left_density = len(left_temps) / candidate_left.size
                    right_density = len(right_temps) / candidate_right.size
                    avg_density = (left_density + right_density) / 2
                    
                    # Combined score (prefer good range and density)
                    score = avg_range * 0.6 + avg_density * 0.4
                    
                    print(f"  Index {i}: temp_range={avg_range:.1f}, density={avg_density:.2f}, score={score:.2f}")
                    
                    if score > best_score:
                        best_score = score
                        best_candidate = (i, candidate_left, candidate_right, total_pixels)
            
            if best_candidate:
                best_idx, best_left, best_right, total_pixels = best_candidate
                print(f"Selected index {best_idx} with score {best_score:.2f} as the best candidate")
                print(f"Left shape: {best_left.shape}, Right shape: {best_right.shape}")
                return best_left, best_right
            else:
                # Fallback to first candidate
                best_idx, best_left, best_right, total_pixels = candidates[0]
                print(f"Using fallback selection: index {best_idx}")
                return best_left, best_right
        
        # Method 2: Try deeper extraction for nested objects
        print("Method 1 failed, trying deeper extraction...")
        for i in range(left_crop.shape[0]):
            try:
                candidate_left = left_crop[i, 0]
                candidate_right = right_crop[i, 0]
                
                # If still objects, try to go deeper
                if hasattr(candidate_left, 'dtype') and candidate_left.dtype == object:
                    if hasattr(candidate_left, 'flat') and candidate_left.size > 0:
                        candidate_left = candidate_left.flat[0]
                        candidate_right = candidate_right.flat[0]
                
                # Convert to numeric arrays
                candidate_left = np.array(candidate_left, dtype=float)
                candidate_right = np.array(candidate_right, dtype=float)
                
                if (len(candidate_left.shape) == 2 and len(candidate_right.shape) == 2 and
                    candidate_left.shape[0] > 50 and candidate_left.shape[1] > 50):
                    
                    print(f"Found valid images at index {i} (deep extraction)")
                    print(f"Left shape: {candidate_left.shape}, Right shape: {candidate_right.shape}")
                    return candidate_left, candidate_right
                    
            except Exception as e:
                continue
    
    # Case 3: Try direct conversion for other structures
    try:
        print("Attempting direct numeric conversion...")
        scan_left = np.array(left_crop, dtype=float)
        scan_right = np.array(right_crop, dtype=float)
        
        if (len(scan_left.shape) == 2 and len(scan_right.shape) == 2 and
            scan_left.shape[0] > 50 and scan_left.shape[1] > 50):
            print("Direct conversion successful")
            return scan_left, scan_right
    except:
        pass
    
    print("Error: Could not extract valid images from any method")
    print("Available data shapes and types:")
    print(f"  Left: shape={left_crop.shape}, dtype={left_crop.dtype}")
    print(f"  Right: shape={right_crop.shape}, dtype={right_crop.dtype}")
    
    # Try to show what's inside object arrays for debugging
    if left_crop.dtype == object:
        print("Object array contents:")
        for i in range(min(3, left_crop.shape[0])):  # Show first 3 elements
            try:
                item = left_crop[i, 0]
                print(f"  Index {i}: type={type(item)}, shape={getattr(item, 'shape', 'N/A')}")
            except:
                print(f"  Index {i}: Could not access")
    
    return None, None

def create_foot_overlay():
    """Create overlay of original left foot with scaled mirrored right foot"""
    warnings.filterwarnings("ignore", category=MatReadWarning)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Processing MAT file: {MAT_FILE}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Load data
    mat = scipy.io.loadmat(MAT_FILE)
    print("Available keys:", list(mat.keys()))
    
    left_crop = mat["Indirect_plantar_Right_crop"]  # Note: naming convention
    right_crop = mat["Indirect_plantar_Left_crop"]
    
    # Use the new detection function
    scan_left, scan_right = detect_and_extract_images(left_crop, right_crop)
    
    if scan_left is None or scan_right is None:
        print("Failed to extract valid images. Aborting.")
        return None
    
    # Process images with adaptive thresholding
    print("Processing left foot with adaptive thresholding...")
    img_left = trim_to_content(to_nan(scan_left, adaptive_threshold=True))
    print("Processing right foot with adaptive thresholding...")
    img_right = trim_to_content(to_nan(scan_right, adaptive_threshold=True))
    img_right_mir = mirror_horiz(img_right)  # Mirror the right foot
    
    print(f"Processed shapes - Left: {img_left.shape}, Right mirrored: {img_right_mir.shape}")
    
    # Optional rotation optimization
    rotation_angle = 0
    overlap_score = 0
    rotation_scores = {}
    
    if ENABLE_ROTATION_OPTIMIZATION:
        print("\n" + "="*50)
        print("ROTATION OPTIMIZATION")
        print("="*50)
        
        # Find optimal rotation angle
        rotation_angle, overlap_score, rotation_scores = find_best_rotation_angle(
            img_left, 
            img_right_mir, 
            angle_range=ROTATION_ANGLE_RANGE, 
            angle_step=ROTATION_ANGLE_STEP
        )
        
        print(f"Best rotation angle: {rotation_angle}° (overlap score: {overlap_score:.3f})")
        
        # Show top 3 angles for reference
        sorted_scores = sorted(rotation_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        print("Top 3 rotation angles:")
        for angle, score in sorted_scores:
            print(f"  {angle:3d}°: {score:.3f}")
        
        # Apply the optimal rotation if it's better than no rotation
        if rotation_angle != 0:
            print(f"Applying {rotation_angle}° rotation to right foot...")
            img_right_mir = rotate_image_preserve_temps(img_right_mir, rotation_angle)
            img_right_mir = trim_to_content(img_right_mir)  # Re-trim after rotation
            print(f"Right foot shape after rotation: {img_right_mir.shape}")
        else:
            print("No rotation needed - 0° was optimal")
        
        print("="*50)
    
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
    titles = ['Light Overlay (alpha=0.3)', 'Medium Overlay (alpha=0.45)', 'Heavy Overlay (alpha=0.6)']
    
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
    
    # Save the visualization with file identifier in filename
    output_path = os.path.join(OUTPUT_DIR, f"foot_overlay_comparison_{SELECTED_FILE}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\\nOverlay visualization saved to: {output_path}")
    
    plt.show()
    
    # Create a focused overlay for analysis
    fig2, ax = plt.subplots(figsize=(10, 8))
    
    # Main overlay
    im_base = ax.imshow(left_canvas, cmap=CMAP)
    im_overlay = ax.imshow(right_canvas, cmap=CMAP, alpha=RIGHT_ALPHA)
    
    if ENABLE_ROTATION_OPTIMIZATION:
        title_text = (f'Foot Overlay Analysis\\nLeft Foot (original) + Right Foot (rotated {rotation_angle}°, scaled alpha={RIGHT_ALPHA})\\n'
                     f'Scale factors: x={scale_x:.3f}, y={scale_y:.3f}, Overlap score: {overlap_score:.3f}')
    else:
        title_text = (f'Foot Overlay Analysis\\nLeft Foot (original) + Right Foot (scaled alpha={RIGHT_ALPHA})\\n'
                     f'Scale factors: x={scale_x:.3f}, y={scale_y:.3f}')
    
    ax.set_title(title_text)
    ax.axis('off')
    
    # Add colorbar
    cbar = plt.colorbar(im_base, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Temperature (°C)')
    
    # Save focused overlay with file identifier in filename
    focused_path = os.path.join(OUTPUT_DIR, f"focused_foot_overlay_{SELECTED_FILE}.png")
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
    if ENABLE_ROTATION_OPTIMIZATION:
        print(f"Rotation angle applied: {rotation_angle}°")
        print(f"Overlap score achieved: {overlap_score:.3f}")
    print(f"Scaling factors applied: x={scale_x:.3f}, y={scale_y:.3f}")
    print(f"Right foot transparency: alpha={RIGHT_ALPHA}")
    
    return {
        'left_original': img_left,
        'right_mirrored': img_right_mir,
        'right_scaled': img_right_scaled,
        'left_canvas': left_canvas,
        'right_canvas': right_canvas,
        'scale_x': scale_x,
        'scale_y': scale_y,
        'rotation_angle': rotation_angle,
        'overlap_score': overlap_score,
        'rotation_scores': rotation_scores
    }

def test_multiple_files():
    """Test the overlay creator with multiple MAT files"""
    global MAT_FILE, OUTPUT_DIR, SELECTED_FILE
    
    results = {}
    for file_key in AVAILABLE_MAT_FILES.keys():
        print(f"\n{'='*80}")
        print(f"TESTING FILE: {file_key}")
        print(f"{'='*80}")
        
        # Update global variables for this file
        SELECTED_FILE = file_key
        MAT_FILE = AVAILABLE_MAT_FILES[file_key]
        OUTPUT_DIR = "output_overlay_system"  # Single output folder for all files
        
        try:
            result = create_foot_overlay()
            results[file_key] = "SUCCESS" if result is not None else "FAILED"
            print(f"\nResult for {file_key}: {results[file_key]}")
        except Exception as e:
            results[file_key] = f"ERROR: {str(e)}"
            print(f"\nResult for {file_key}: {results[file_key]}")
    
    print(f"\n{'='*80}")
    print("SUMMARY OF ALL FILES:")
    print(f"{'='*80}")
    for file_key, result in results.items():
        print(f"{file_key:10} : {result}")

if __name__ == "__main__":
    # Single file processing
    print("=== SINGLE FILE MODE ===")
    results = create_foot_overlay()
    
    # Uncomment the lines below to test all files at once
    # print("\n=== TESTING ALL AVAILABLE FILES ===")
    # test_multiple_files()