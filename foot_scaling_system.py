# foot_scaling_system.py
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.io.matlab import MatReadWarning
from scipy.ndimage import zoom
import matplotlib.patches as patches

# ==========================
# CONFIGURATION
# ==========================
MAT_FILE = "Data/Temp Data/pnt_mat_files/pnt1.mat"
DAY_INDEX = 0
OUTPUT_DIR = "output_scaling_system"
CMAP = "hot"
RIGHT_ALPHA = 0.45  # Transparency for overlay

# ==========================
# FOOT SCALER CLASS
# ==========================
class FootScaler:
    def __init__(self, foot_image, name):
        self.original_image = foot_image.copy()
        self.name = name
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.scaled_image = None
        
    def scale_to_match(self, target_foot):
        """Scale this foot to match the target foot's dimensions"""
        target_h, target_w = target_foot.shape
        current_h, current_w = self.original_image.shape
        
        self.scale_x = target_w / current_w
        self.scale_y = target_h / current_h
        
        print(f"Scaling {self.name}: scale_x={self.scale_x:.3f}, scale_y={self.scale_y:.3f}")
        
        # Use zoom for scaling while preserving temperature data
        self.scaled_image = self._scale_image_preserve_temps(
            self.original_image, self.scale_x, self.scale_y
        )
        
        return self.scaled_image
    
    def _scale_image_preserve_temps(self, img, scale_x, scale_y):
        """Scale image while preserving temperature values using careful interpolation"""
        # Create mask of valid pixels
        valid_mask = ~np.isnan(img)
        
        # Replace NaN with a safe value for scaling
        min_temp = np.nanmin(img)
        if np.isnan(min_temp):
            fill_value = 20.0  # Default room temperature
        else:
            fill_value = min_temp - 5.0
        
        img_filled = np.where(valid_mask, img, fill_value)
        
        # Scale the image and mask separately
        scaled_img = zoom(img_filled, (scale_y, scale_x), order=1, mode='nearest')
        scaled_mask = zoom(valid_mask.astype(float), (scale_y, scale_x), order=1, mode='nearest')
        
        # Restore NaN values where mask indicates invalid pixels
        final_img = np.where(scaled_mask > 0.5, scaled_img, np.nan)
        
        return final_img
    
    def get_scaled_image(self):
        """Get the scaled image"""
        if self.scaled_image is None:
            return self.original_image
        return self.scaled_image
    
    def transform_point_to_scaled(self, x, y):
        """Transform original coordinates to scaled coordinates"""
        scaled_x = x * self.scale_x
        scaled_y = y * self.scale_y
        return scaled_x, scaled_y
    
    def transform_point_to_original(self, scaled_x, scaled_y):
        """Transform scaled coordinates back to original coordinates"""
        original_x = scaled_x / self.scale_x
        original_y = scaled_y / self.scale_y
        return original_x, original_y

# ==========================
# COORDINATE TRANSFORMATION FUNCTIONS
# ==========================
def calculate_scaling_factors(small_img, large_img):
    """Calculate scaling factors to make small_img match large_img dimensions"""
    large_h, large_w = large_img.shape
    small_h, small_w = small_img.shape
    
    scale_x = large_w / small_w
    scale_y = large_h / small_h
    
    return scale_x, scale_y

def transform_coordinate_to_scaled(x, y, scale_x, scale_y):
    """Convert original coordinate to scaled coordinate"""
    return x * scale_x, y * scale_y

def transform_coordinate_to_original(scaled_x, scaled_y, scale_x, scale_y):
    """Convert scaled coordinate back to original"""
    return scaled_x / scale_x, scaled_y / scale_y

# ==========================
# INTERACTIVE MARKING SYSTEM
# ==========================
class InteractiveMarker:
    def __init__(self, left_foot_scaler, right_foot_scaler):
        self.left_scaler = left_foot_scaler
        self.right_scaler = right_foot_scaler
        self.marked_points = []
        self.fig = None
        self.axes = None
        
    def create_marking_interface(self):
        """Create interactive interface for marking points on both feet"""
        # Get images to display (both should be same size now)
        left_display = self.left_scaler.get_scaled_image()
        right_display = self.right_scaler.get_scaled_image()
        
        # Create side-by-side display
        self.fig, self.axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Display left foot
        im1 = self.axes[0].imshow(left_display, cmap=CMAP)
        self.axes[0].set_title(f'{self.left_scaler.name} (for marking)')
        self.axes[0].axis('off')
        
        # Display right foot
        im2 = self.axes[1].imshow(right_display, cmap=CMAP)
        self.axes[1].set_title(f'{self.right_scaler.name} (for marking)')
        self.axes[1].axis('off')
        
        # Add colorbars
        plt.colorbar(im1, ax=self.axes[0], fraction=0.046, pad=0.04)
        plt.colorbar(im2, ax=self.axes[1], fraction=0.046, pad=0.04)
        
        # Connect click event
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        plt.tight_layout()
        return self.fig
    
    def on_click(self, event):
        """Handle mouse click events for marking points"""
        if event.inaxes is None:
            return
        
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        
        # Determine which foot was clicked
        if event.inaxes == self.axes[0]:
            foot_side = 'left'
            scaler = self.left_scaler
        elif event.inaxes == self.axes[1]:
            foot_side = 'right'
            scaler = self.right_scaler
        else:
            return
        
        # Store the point
        point_info = {
            'foot': foot_side,
            'scaled_x': x,
            'scaled_y': y,
            'scaler': scaler
        }
        
        # Calculate original coordinates
        orig_x, orig_y = scaler.transform_point_to_original(x, y)
        point_info['original_x'] = orig_x
        point_info['original_y'] = orig_y
        
        self.marked_points.append(point_info)
        
        # Visualize the marked point
        circle = plt.Circle((x, y), radius=3, color='lime', fill=True)
        event.inaxes.add_patch(circle)
        
        # Add point number
        point_num = len(self.marked_points)
        event.inaxes.text(x + 5, y - 5, str(point_num), color='lime', fontsize=12, fontweight='bold')
        
        self.fig.canvas.draw()
        
        print(f"Point {point_num} marked on {foot_side} foot:")
        print(f"  Scaled coordinates: ({x:.1f}, {y:.1f})")
        print(f"  Original coordinates: ({orig_x:.1f}, {orig_y:.1f})")
        print()
    
    def get_marked_points(self):
        """Get all marked points"""
        return self.marked_points
    
    def clear_points(self):
        """Clear all marked points"""
        self.marked_points = []
        # Redraw the interface
        if self.fig:
            for ax in self.axes:
                for patch in ax.patches[:]:
                    patch.remove()
                for text in ax.texts[:]:
                    text.remove()
            self.fig.canvas.draw()

# ==========================
# HELPER FUNCTIONS FROM ALIGNING_VERTICALLY
# ==========================
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
    """Trim rows/cols that are entirely NaN."""
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
    """Mirror horizontally (left-right flip)."""
    if img is None or img.ndim < 2:
        return img
    return np.fliplr(img)

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

def create_overlay_visualization(left_original, right_scaled_mirrored, output_dir, suffix=""):
    """Create overlay visualization of original left foot with scaled mirrored right foot"""
    
    # Ensure both images have the same dimensions for overlay
    H = max(left_original.shape[0], right_scaled_mirrored.shape[0])
    W = max(left_original.shape[1], right_scaled_mirrored.shape[1])
    
    left_canvas = pad_to_canvas_centered(left_original, H, W)
    right_canvas = pad_to_canvas_centered(right_scaled_mirrored, H, W)
    
    # Create overlay visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original left foot
    im1 = axes[0].imshow(left_canvas, cmap=CMAP)
    axes[0].set_title('Original Left Foot')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Scaled mirrored right foot
    im2 = axes[1].imshow(right_canvas, cmap=CMAP)
    axes[1].set_title('Right Foot (Scaled & Mirrored)')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Overlay
    axes[2].imshow(left_canvas, cmap=CMAP)  # Base layer
    axes[2].imshow(right_canvas, cmap=CMAP, alpha=RIGHT_ALPHA)  # Overlay
    axes[2].set_title(f'Overlay: Left (base) + Right (α={RIGHT_ALPHA})')
    axes[2].axis('off')
    
    # Add colorbar for the overlay
    im3 = axes[2].imshow(left_canvas, cmap=CMAP)
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    # Save the overlay
    output_filename = f"foot_overlay_comparison{suffix}.png"
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print(f"Overlay visualization saved to: {output_path}")
    
    plt.show()
    
    return output_path

# ==========================
# MAIN FUNCTION
# ==========================
def main():
    warnings.filterwarnings("ignore", category=MatReadWarning)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load MATLAB data
    mat = scipy.io.loadmat(MAT_FILE)
    
    print("Available keys in MAT file:", list(mat.keys()))
    
    # Load foot data
    if "Indirect_plantar_Right_crop" not in mat or "Indirect_plantar_Left_crop" not in mat:
        print("Required keys not found in MAT file")
        return
    
    left_crop = mat["Indirect_plantar_Right_crop"]  # Note: naming convention
    right_crop = mat["Indirect_plantar_Left_crop"]
    
    print(f"Left crop shape: {left_crop.shape}")
    print(f"Right crop shape: {right_crop.shape}")
    
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
        # Direct 2D image data or other structure
        if len(left_crop.shape) == 2:
            scan_left = left_crop
            scan_right = right_crop
        else:
            scan_left = left_crop
            scan_right = right_crop
    
    # Process images
    img_left = trim_to_content(to_nan(scan_left))
    img_right = trim_to_content(to_nan(scan_right))
    img_right_mir = mirror_horiz(img_right)  # Mirror right foot
    
    print(f"Processed left foot shape: {img_left.shape}")
    print(f"Processed right foot (mirrored) shape: {img_right_mir.shape}")
    
    # Determine which foot is smaller
    left_size = img_left.shape[0] * img_left.shape[1]
    right_size = img_right_mir.shape[0] * img_right_mir.shape[1]
    
    print(f"Left foot size: {left_size} pixels")
    print(f"Right foot size: {right_size} pixels")
    
    # Create FootScaler objects
    if left_size < right_size:
        print("\\nLeft foot is smaller - scaling it to match right foot")
        smaller_scaler = FootScaler(img_left, "Left Foot (scaled)")
        larger_scaler = FootScaler(img_right_mir, "Right Foot (original)")
        smaller_scaler.scale_to_match(img_right_mir)
        
        # For overlay: use original left and scaled right
        overlay_left = img_left  # Original left
        overlay_right = img_right_mir  # Original right (already mirrored)
    else:
        print("\\nRight foot is smaller - scaling it to match left foot")
        smaller_scaler = FootScaler(img_right_mir, "Right Foot (scaled)")
        larger_scaler = FootScaler(img_left, "Left Foot (original)")
        smaller_scaler.scale_to_match(img_left)
        
        # For overlay: use original left and scaled right
        overlay_left = img_left  # Original left
        overlay_right = smaller_scaler.get_scaled_image()  # Scaled right
    
    # Create overlay visualization
    print("\\nCreating overlay visualization...")
    create_overlay_visualization(overlay_left, overlay_right, OUTPUT_DIR)
    
    # Create interactive marking system
    print("\\nCreating interactive marking interface...")
    print("Instructions:")
    print("1. Click on corresponding points on both feet")
    print("2. Points will be numbered and shown in green")
    print("3. Original coordinates are automatically calculated")
    print("4. Close the plot window when done marking")
    print()
    
    marker = InteractiveMarker(smaller_scaler, larger_scaler)
    fig = marker.create_marking_interface()
    
    plt.show()
    
    # After marking, show results
    marked_points = marker.get_marked_points()
    if marked_points:
        print(f"\\n=== MARKED POINTS SUMMARY ===")
        for i, point in enumerate(marked_points, 1):
            print(f"Point {i} ({point['foot']} foot):")
            print(f"  Scaled coordinates: ({point['scaled_x']:.1f}, {point['scaled_y']:.1f})")
            print(f"  Original coordinates: ({point['original_x']:.1f}, {point['original_y']:.1f})")
            print(f"  Scale factors: x={point['scaler'].scale_x:.3f}, y={point['scaler'].scale_y:.3f}")
            print()
    else:
        print("No points were marked.")

if __name__ == "__main__":
    main()