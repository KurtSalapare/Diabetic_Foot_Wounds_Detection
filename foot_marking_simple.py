# foot_marking_simple.py
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
OUTPUT_DIR = "output_marking"
CMAP = "hot"

# Global variables for point collection
marked_points = []
current_point_id = 1

def scale_image_preserve_temps(img, scale_x, scale_y):
    """Scale image while preserving temperature values"""
    valid_mask = ~np.isnan(img)
    min_temp = np.nanmin(img)
    fill_value = min_temp - 5.0 if not np.isnan(min_temp) else 20.0
    
    img_filled = np.where(valid_mask, img, fill_value)
    scaled_img = zoom(img_filled, (scale_y, scale_x), order=1, mode='nearest')
    scaled_mask = zoom(valid_mask.astype(float), (scale_y, scale_x), order=1, mode='nearest')
    
    return np.where(scaled_mask > 0.5, scaled_img, np.nan)

def transform_to_original(scaled_x, scaled_y, scale_x, scale_y):
    """Transform scaled coordinates back to original"""
    return scaled_x / scale_x, scaled_y / scale_y

def on_click(event, scale_info):
    """Handle mouse click events"""
    global marked_points, current_point_id
    
    if event.inaxes is None or event.xdata is None or event.ydata is None:
        return
    
    x, y = event.xdata, event.ydata
    
    # Determine which foot was clicked
    foot_side = 'left' if event.inaxes.get_title().startswith('Left') else 'right'
    
    # Calculate original coordinates if this is the scaled foot
    if 'scaled' in event.inaxes.get_title().lower():
        orig_x, orig_y = transform_to_original(x, y, scale_info['scale_x'], scale_info['scale_y'])
    else:
        orig_x, orig_y = x, y
    
    # Store point information
    point_info = {
        'id': current_point_id,
        'foot': foot_side,
        'display_x': x,
        'display_y': y,
        'original_x': orig_x,
        'original_y': orig_y
    }
    marked_points.append(point_info)
    
    # Visualize the point
    circle = plt.Circle((x, y), radius=3, color='lime', fill=True)
    event.inaxes.add_patch(circle)
    event.inaxes.text(x + 5, y - 5, str(current_point_id), color='lime', fontsize=12, fontweight='bold')
    
    plt.draw()
    
    print(f"\\nPoint {current_point_id} marked on {foot_side} foot:")
    print(f"  Display coordinates: ({x:.1f}, {y:.1f})")
    print(f"  Original coordinates: ({orig_x:.1f}, {orig_y:.1f})")
    
    current_point_id += 1

def create_marking_interface():
    """Create the interactive marking interface"""
    warnings.filterwarnings("ignore", category=MatReadWarning)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load and process data
    mat = scipy.io.loadmat(MAT_FILE)
    left_crop = mat["Indirect_plantar_Right_crop"]
    right_crop = mat["Indirect_plantar_Left_crop"]
    
    print(f"Raw data shapes - Left: {left_crop.shape}, Right: {right_crop.shape}")
    
    # Helper functions
    def to_nan(img):
        """
        Convert temperature values to NaN where appropriate.
        Handles different data structures including object arrays.
        """
        if img is None:
            return img
        if np.isscalar(img):
            return np.nan if img == 0 else img
        
        # Handle object arrays by finding the first valid 2D array
        if hasattr(img, 'dtype') and img.dtype == object:
            if hasattr(img, 'shape') and len(img.shape) == 2 and img.shape[1] == 1:
                # Try each row to find a valid 2D temperature array
                for i in range(img.shape[0]):
                    try:
                        candidate = img[i, 0]
                        if hasattr(candidate, 'shape') and len(candidate.shape) == 2:
                            img = candidate
                            break
                    except:
                        continue
                else:
                    return img  # No valid array found, return as-is
            else:
                return img  # Unexpected structure, return as-is
        
        img = np.array(img)
        if img.size == 0:
            return img
        
        # Convert low temperature values to NaN (background removal)
        if np.issubdtype(img.dtype, np.number):
            return np.where(img < 25, np.nan, img)  # Use 25°C as threshold for thermal data
        else:
            return np.where(img == 0, np.nan, img)  # Use 0 for other data types
    
    def trim_to_content(img):
        if img.ndim < 2: return img
        valid_rows = ~np.all(np.isnan(img), axis=1)
        valid_cols = ~np.all(np.isnan(img), axis=0)
        return img[valid_rows][:, valid_cols] if np.any(valid_rows) and np.any(valid_cols) else img
    
    # Handle different data structures
    if left_crop.shape == (10, 1) and left_crop.dtype == object:
        # Extract first valid image from object arrays
        print("Detected object array structure - extracting first valid image")
        scan_left = scan_right = None
        for i in range(left_crop.shape[0]):
            try:
                candidate_left = left_crop[i, 0]
                candidate_right = right_crop[i, 0]
                if (hasattr(candidate_left, 'shape') and len(candidate_left.shape) == 2 and 
                    hasattr(candidate_right, 'shape') and len(candidate_right.shape) == 2):
                    scan_left = candidate_left
                    scan_right = candidate_right
                    print(f"Using images from index {i}")
                    break
            except:
                continue
        if scan_left is None:
            print("Error: Could not extract valid images")
            return
    else:
        scan_left = left_crop
        scan_right = right_crop
    
    img_left = trim_to_content(to_nan(scan_left))
    img_right = trim_to_content(to_nan(np.fliplr(scan_right)))  # Mirror right foot
    
    print(f"Left foot shape: {img_left.shape}")
    print(f"Right foot shape: {img_right.shape}")
    
    # Determine scaling
    left_size = img_left.shape[0] * img_left.shape[1]
    right_size = img_right.shape[0] * img_right.shape[1]
    
    if left_size < right_size:
        # Scale left foot
        scale_x = img_right.shape[1] / img_left.shape[1]
        scale_y = img_right.shape[0] / img_left.shape[0]
        left_display = scale_image_preserve_temps(img_left, scale_x, scale_y)
        right_display = img_right
        scale_info = {'scale_x': scale_x, 'scale_y': scale_y, 'scaled_foot': 'left'}
        print(f"\\nScaling left foot by factors: x={scale_x:.3f}, y={scale_y:.3f}")
    else:
        # Scale right foot
        scale_x = img_left.shape[1] / img_right.shape[1]
        scale_y = img_left.shape[0] / img_right.shape[0]
        left_display = img_left
        right_display = scale_image_preserve_temps(img_right, scale_x, scale_y)
        scale_info = {'scale_x': scale_x, 'scale_y': scale_y, 'scaled_foot': 'right'}
        print(f"\\nScaling right foot by factors: x={scale_x:.3f}, y={scale_y:.3f}")
    
    # Create the interface
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    im1 = ax1.imshow(left_display, cmap=CMAP)
    title1 = 'Left Foot (scaled)' if scale_info['scaled_foot'] == 'left' else 'Left Foot (original)'
    ax1.set_title(title1)
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1)
    
    im2 = ax2.imshow(right_display, cmap=CMAP)
    title2 = 'Right Foot (scaled)' if scale_info['scaled_foot'] == 'right' else 'Right Foot (original)'
    ax2.set_title(title2)
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2)
    
    # Connect click event
    fig.canvas.mpl_connect('button_press_event', lambda event: on_click(event, scale_info))
    
    plt.tight_layout()
    
    print("\\n" + "="*60)
    print("INTERACTIVE FOOT MARKING SYSTEM")
    print("="*60)
    print("Instructions:")
    print("1. Click on corresponding points on both feet")
    print("2. Points will appear as green circles with numbers")
    print("3. Original coordinates are automatically calculated")
    print("4. Close the window when done marking")
    print("5. Results will be printed below")
    print("="*60)
    
    plt.show()
    
    # Print final results
    if marked_points:
        print("\\n" + "="*60)
        print("MARKED POINTS SUMMARY")
        print("="*60)
        for point in marked_points:
            scaled_info = " (on scaled foot)" if ((point['foot'] == 'left' and scale_info['scaled_foot'] == 'left') or 
                                                 (point['foot'] == 'right' and scale_info['scaled_foot'] == 'right')) else ""
            print(f"Point {point['id']} - {point['foot'].title()} Foot{scaled_info}:")
            print(f"  Display: ({point['display_x']:.1f}, {point['display_y']:.1f})")
            print(f"  Original: ({point['original_x']:.1f}, {point['original_y']:.1f})")
            print()
        
        # Save results to file
        with open(os.path.join(OUTPUT_DIR, "marked_points.txt"), "w") as f:
            f.write("FOOT MARKING RESULTS\\n")
            f.write("="*50 + "\\n")
            f.write(f"Scaling information: {scale_info['scaled_foot']} foot was scaled\\n")
            f.write(f"Scale factors: x={scale_info['scale_x']:.3f}, y={scale_info['scale_y']:.3f}\\n\\n")
            
            for point in marked_points:
                f.write(f"Point {point['id']} ({point['foot']} foot):\\n")
                f.write(f"  Display coordinates: ({point['display_x']:.1f}, {point['display_y']:.1f})\\n")
                f.write(f"  Original coordinates: ({point['original_x']:.1f}, {point['original_y']:.1f})\\n\\n")
        
        print(f"Results saved to: {os.path.join(OUTPUT_DIR, 'marked_points.txt')}")
    else:
        print("\\nNo points were marked.")

if __name__ == "__main__":
    create_marking_interface()