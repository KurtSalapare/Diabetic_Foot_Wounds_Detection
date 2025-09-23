# overlay_mirror_align_by_max_y.py
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.io.matlab import MatReadWarning

# ==========================
# CONFIGURATION
# ==========================
MAT_FILE   = "Data/Temp Data/pnt_mat_files/pnt2.mat"
DAY_INDEX  = 0                   # 0-based: 0 = Day 1
OUTPUT_DIR = "output_overlay_maxY"
OUTPUT_PNG = f"overlay_day{DAY_INDEX+1}_mirror_align_maxY.png"
CMAP       = "hot"
RIGHT_ALPHA = 0.45               # overlay transparency for mirrored right foot

# ==========================
# HELPERS
# ==========================
def to_nan(img):
    """Convert 0 to NaN (treat 0 as background)."""
    return np.where(img == 0, np.nan, img)

def trim_to_content(img):
    """Trim rows/cols that are entirely NaN."""
    if img is None or img.size == 0:
        return img
    
    # Ensure img is at least 2D
    if img.ndim == 0:
        print(f"Warning: Got 0-dimensional array, returning as-is")
        return img
    elif img.ndim == 1:
        print(f"Warning: Got 1-dimensional array with shape {img.shape}, reshaping to 2D")
        img = img.reshape(-1, 1)
    
    valid_rows = ~(np.all(np.isnan(img), axis=1))
    valid_cols = ~(np.all(np.isnan(img), axis=0))
    if not np.any(valid_rows) or not np.any(valid_cols):
        return img  # nothing valid; return as-is
    return img[valid_rows][:, valid_cols]

def mirror_horiz(img):
    """Mirror horizontally (left-right flip)."""
    if img is None:
        return img
    if img.ndim < 2:
        print(f"Warning: Cannot mirror {img.ndim}D array, need at least 2D")
        return img
    return np.fliplr(img)

def max_valid_row_index(img):
    """
    Return the largest row index (max-y in image coordinates)
    that contains at least one non-NaN value.
    If none found, return -1.
    """
    valid_rows = np.any(~np.isnan(img), axis=1)
    idx = np.where(valid_rows)[0]
    return int(idx.max()) if idx.size > 0 else -1

def find_horizontal_center(img):
    """
    Find the horizontal center (x-coordinate) of all valid (non-NaN) pixels.
    Returns the mean x-coordinate of valid pixels, or -1 if no valid pixels found.
    """
    valid_mask = ~np.isnan(img)
    y_coords, x_coords = np.where(valid_mask)
    
    if len(x_coords) == 0:
        return -1
    
    return np.mean(x_coords)

def pad_to_canvas_centered(img, H, W):
    """
    Center 'img' inside a (H, W) canvas (NaN background).
    Horizontal & vertical centering.
    """
    canvas = np.full((H, W), np.nan, dtype=float)
    h, w = img.shape
    top  = max(0, (H - h) // 2)
    left = max(0, (W - w) // 2)
    canvas[top:top+h, left:left+w] = img
    return canvas

def horizontal_align_by_center(base_img, move_img):
    """
    Horizontally align 'move_img' to 'base_img' so their horizontal centers
    (centroid of valid pixels) coincide. Returns horizontally-shifted 'move_img'.
    """
    # Find horizontal centers
    x_base = find_horizontal_center(base_img)
    x_move = find_horizontal_center(move_img)
    
    if x_base < 0 or x_move < 0:
        # One (or both) are empty—return move_img as-is
        print("Warning: Cannot find horizontal centers for alignment")
        return move_img
    
    print(f"Base image horizontal center: {x_base:.1f}, Move image horizontal center: {x_move:.1f}")
    
    # Calculate required horizontal shift
    h_m, w_m = move_img.shape
    shift_x = int(round(x_base - x_move))
    
    print(f"Horizontal shift needed: {shift_x} pixels")
    
    if shift_x == 0:
        return move_img
    
    # Apply horizontal shift by padding
    if shift_x > 0:
        # Shift right: add padding on the left
        pad_left = shift_x
        pad_right = 0
    else:
        # Shift left: add padding on the right
        pad_left = 0
        pad_right = -shift_x
    
    aligned = np.pad(
        move_img,
        ((0, 0), (pad_left, pad_right)),
        mode="constant",
        constant_values=np.nan
    )
    return aligned

def align_both_axes(base_img, move_img):
    """
    Align move_img to base_img both vertically (by max-y) and horizontally (by center).
    """
    # First apply vertical alignment
    move_v_aligned = vertical_align_by_max_y(base_img, move_img)
    
    # Then apply horizontal alignment
    move_vh_aligned = horizontal_align_by_center(base_img, move_v_aligned)
    
    return move_vh_aligned

def vertical_align_by_max_y(base_img, move_img):
    """
    Vertically align 'move_img' to 'base_img' so their max-y (bottom-most)
    valid rows coincide. Returns vertically-shifted 'move_img' on a new canvas.

    We only apply a vertical shift (by padding at top/bottom), no resampling.
    Width is unchanged here; caller can later center/pad both to common canvas.
    """
    # Identify bottom-most valid rows
    y_base = max_valid_row_index(base_img)
    y_move = max_valid_row_index(move_img)

    if y_base < 0 or y_move < 0:
        # One (or both) are empty—return move_img as-is
        return move_img

    print(f"Base image max-y row: {y_base}, Move image max-y row: {y_move}")

    # Compute required shift: want y_move_shifted == y_base
    # If y_move < y_base, add TOP padding to move_img
    # If y_move > y_base, add BOTTOM padding to move_img
    h_m, w_m = move_img.shape
    if y_move < y_base:
        pad_top = (y_base - y_move)
        pad_bottom = 0
        print(f"Adding {pad_top} pixels of top padding for vertical alignment")
    else:
        pad_top = 0
        pad_bottom = (y_move - y_base)
        print(f"Adding {pad_bottom} pixels of bottom padding for vertical alignment")

    aligned = np.pad(
        move_img,
        ((pad_top, pad_bottom), (0, 0)),
        mode="constant",
        constant_values=np.nan
    )
    return aligned
    """
    Vertically align 'move_img' to 'base_img' so their max-y (bottom-most)
    valid rows coincide. Returns vertically-shifted 'move_img' on a new canvas.

    We only apply a vertical shift (by padding at top/bottom), no resampling.
    Width is unchanged here; caller can later center/pad both to common canvas.
    """
    # Identify bottom-most valid rows
    y_base = max_valid_row_index(base_img)
    y_move = max_valid_row_index(move_img)

    if y_base < 0 or y_move < 0:
        # One (or both) are empty—return move_img as-is
        return move_img

    # Compute required shift: want y_move_shifted == y_base
    # If y_move < y_base, add TOP padding to move_img
    # If y_move > y_base, add BOTTOM padding to move_img
    h_m, w_m = move_img.shape
    if y_move < y_base:
        pad_top = (y_base - y_move)
        pad_bottom = 0
    else:
        pad_top = 0
        pad_bottom = (y_move - y_base)

    aligned = np.pad(
        move_img,
        ((pad_top, pad_bottom), (0, 0)),
        mode="constant",
        constant_values=np.nan
    )
    return aligned

# ==========================
# MAIN
# ==========================
def main():
    warnings.filterwarnings("ignore", category=MatReadWarning)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load MATLAB data
    mat = scipy.io.loadmat(MAT_FILE)
    
    # Debug: Print available keys and their shapes
    print("Available keys in MAT file:", list(mat.keys()))
    
    # NOTE: Based on earlier datasets:
    # - "Indirect_plantar_Right_crop" contains LEFT foot scans
    # - "Indirect_plantar_Left_crop"  contains RIGHT foot scans
    if "Indirect_plantar_Right_crop" in mat:
        left_crop = mat["Indirect_plantar_Right_crop"]
        print(f"Left crop shape: {left_crop.shape}")
    else:
        print("Warning: 'Indirect_plantar_Right_crop' not found in MAT file")
        return
    
    if "Indirect_plantar_Left_crop" in mat:
        right_crop = mat["Indirect_plantar_Left_crop"]
        print(f"Right crop shape: {right_crop.shape}")
    else:
        print("Warning: 'Indirect_plantar_Left_crop' not found in MAT file")
        return

    # Check if day index is valid
    if DAY_INDEX >= left_crop.shape[0] or DAY_INDEX >= right_crop.shape[0]:
        print(f"Error: DAY_INDEX {DAY_INDEX} is out of bounds for arrays with {left_crop.shape[0]} and {right_crop.shape[0]} days")
        return
    
    # The data structure appears to be different - let's try direct access first
    # If the arrays are already 2D images, use them directly
    if len(left_crop.shape) == 2 and len(right_crop.shape) == 2:
        scan_left = left_crop
        scan_right = right_crop
        print(f"Using direct 2D arrays - Left: {scan_left.shape}, Right: {scan_right.shape}")
    else:
        # Try the original indexing method if it's a 3D array
        try:
            scan_left = left_crop[DAY_INDEX, 0] if left_crop.ndim > 2 else left_crop
            scan_right = right_crop[DAY_INDEX, 0] if right_crop.ndim > 2 else right_crop
        except (IndexError, TypeError) as e:
            print(f"Error accessing data with indexing: {e}")
            # Fallback: use the arrays directly
            scan_left = left_crop
            scan_right = right_crop
    
    print(f"Final scan left shape: {scan_left.shape}, type: {type(scan_left)}")
    print(f"Final scan right shape: {scan_right.shape}, type: {type(scan_right)}")
    
    # Check if scans are empty or malformed
    if scan_left.size == 0:
        print("Error: Left scan is empty")
        return
    if scan_right.size == 0:
        print("Error: Right scan is empty")
        return

    # Convert zeros to NaNs, then trim borders to content
    img_left  = trim_to_content(to_nan(scan_left))
    img_right = trim_to_content(to_nan(scan_right))

    # Mirror the RIGHT foot horizontally so curvature matches LEFT
    img_right_mir = mirror_horiz(img_right)

    # -------- Vertical and horizontal alignment --------
    # We align RIGHT (mirrored) to LEFT both vertically (by max-y) and horizontally (by center)
    print("Performing vertical and horizontal alignment...")
    img_right_mir_aligned = align_both_axes(img_left, img_right_mir)

    # -------- Put both on a common canvas, centered --------
    H = max(img_left.shape[0], img_right_mir_aligned.shape[0])
    W = max(img_left.shape[1], img_right_mir_aligned.shape[1])

    left_canvas  = pad_to_canvas_centered(img_left, H, W)
    right_canvas = pad_to_canvas_centered(img_right_mir_aligned, H, W)

    # -------- Overlay visualization --------
    fig, ax = plt.subplots(figsize=(10, 6))

    # Left foot as base
    im1 = ax.imshow(left_canvas, cmap=CMAP)
    # Mirrored+aligned right foot on top
    im2 = ax.imshow(right_canvas, cmap=CMAP, alpha=RIGHT_ALPHA)

    ax.set_title(f"Day {DAY_INDEX+1}: Mirrored Right Overlaid on Left\n"
                 f"(Vertical alignment by max-y + Horizontal alignment by center)")
    ax.axis("off")

    # Single colorbar tied to the base image
    cbar = plt.colorbar(im1, ax=ax, fraction=0.035, pad=0.04)
    cbar.set_label("Temperature (°C)")

    out_path = os.path.join(OUTPUT_DIR, OUTPUT_PNG)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved overlay to: {out_path}")

if __name__ == "__main__":
    main()
