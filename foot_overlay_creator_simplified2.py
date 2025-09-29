import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.ndimage import zoom, rotate, map_coordinates
import torch

# Configuration
AVAILABLE_MAT_FILES = {
    "gz1": "Data/Temp Data/gz1.mat",
    "gz2": "Data/Temp Data/gz2.mat",
    "gz3": "Data/Temp Data/gz3.mat",
    "gz4": "Data/Temp Data/gz4.mat",
    "gz7": "Data/Temp Data/gz7.mat",
    "gz8": "Data/Temp Data/gz8.mat",
    "gz9": "Data/Temp Data/gz9.mat",
    "pnt1": "Data/Temp Data/pnt_mat_files/pnt1.mat",
    "pnt2": "Data/Temp Data/pnt_mat_files/pnt2.mat",
    "pnt3": "Data/Temp Data/pnt_mat_files/pnt3.mat",
}
SELECTED_FILE = "gz3"
MAT_FILE = AVAILABLE_MAT_FILES[SELECTED_FILE]
OUTPUT_DIR = "output_overlay_system"
CMAP = "hot"
RIGHT_ALPHA = 0.45
ENABLE_ROTATION_OPTIMIZATION = True
ROTATION_ANGLE_RANGE = (-30, 30)
ROTATION_ANGLE_STEP = 0.5
ENABLE_FOCUSED_OVERLAY = False
MANUAL_INDEX_SELECTION = None
ENABLE_WARP_OPTIMIZATION = True

def scale_image(img, scale_x, scale_y):
    if not np.any(~np.isnan(img)):
        return np.full((int(img.shape[0] * scale_y), int(img.shape[1] * scale_x)), np.nan)
    
    y_new, x_new = np.mgrid[0:int(img.shape[0] * scale_y), 0:int(img.shape[1] * scale_x)]
    scaled_img = map_coordinates(img, [y_new / scale_y, x_new / scale_x], order=0, mode='constant', cval=np.nan, prefilter=False).reshape(y_new.shape)
    return scaled_img

def to_nan(img, adaptive_threshold=True):
    if img is None or np.isscalar(img):
        return np.nan if img is None or img < 25 else img
    
    img = np.array(img, dtype=float) if not isinstance(img, np.ndarray) else img
    if img.size == 0:
        return img
    
    if adaptive_threshold:
        valid_data = img[~np.isnan(img)]
        if len(valid_data) > 0:
            min_temp, max_temp = np.min(valid_data), np.max(valid_data)
            if max_temp - min_temp > 5 and 15 < min_temp < 50:
                return np.where(img < min_temp + 0.05 * (max_temp - min_temp), np.nan, img)
    return np.where(img == 0, np.nan, img)

def trim_to_content(img):
    if img is None or not isinstance(img, np.ndarray) or img.ndim < 2 or img.size == 0:
        return img
    valid_rows = ~np.all(np.isnan(img), axis=1)
    valid_cols = ~np.all(np.isnan(img), axis=0)
    return img[valid_rows][:, valid_cols] if np.any(valid_rows) and np.any(valid_cols) else img

def mirror_horiz(img):
    return np.fliplr(img) if img is not None and img.ndim >= 2 else img

def pad_to_same_size(img1, img2):
    H, W = max(img1.shape[0], img2.shape[0]), max(img1.shape[1], img2.shape[1])
    canvas1 = np.full((H, W), np.nan, dtype=float)
    canvas2 = np.full((H, W), np.nan, dtype=float)
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    canvas1[(H-h1)//2:(H-h1)//2+h1, (W-w1)//2:(W-w1)//2+w1] = img1
    canvas2[(H-h2)//2:(H-h2)//2+h2, (W-w2)//2:(W-w2)//2+w2] = img2
    return canvas1, canvas2

def rotate_image(img, angle):
    if angle == 0:
        return img
    fill_value = np.nanmin(img) - 10.0 if not np.isnan(np.nanmin(img)) else 15.0
    img_filled = np.where(~np.isnan(img), img, fill_value)
    rotated_img = rotate(img_filled, angle, order=0, mode='constant', cval=fill_value, reshape=True)
    rotated_mask = rotate((~np.isnan(img)).astype(float), angle, order=0, mode='constant', cval=0, reshape=True)
    return np.where(rotated_mask > 0.5, rotated_img, np.nan)

def create_binary_mask(img, adaptive_threshold=True):
    if img is None:
        return None
    valid_data = img[~np.isnan(img)]
    if len(valid_data) == 0:
        return np.zeros_like(img, dtype=bool)
    threshold = np.min(valid_data) + 0.05 * (np.max(valid_data) - np.min(valid_data)) if adaptive_threshold and len(valid_data) > 0 else 25
    return (~np.isnan(img)) & (img > threshold)

def calculate_overlap_score(mask1, mask2):
    if mask1.shape != mask2.shape:
        return 0
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0

def find_best_rotation_angle(left_foot, right_foot_mirrored):
    left_mask = create_binary_mask(left_foot)
    best_angle, best_score, scores = 0, 0, {}
    # Use numpy.arange to handle float step sizes
    angles = np.arange(ROTATION_ANGLE_RANGE[0], ROTATION_ANGLE_RANGE[1] + ROTATION_ANGLE_STEP, ROTATION_ANGLE_STEP)
    for angle in angles:
        rotated_right = rotate_image(right_foot_mirrored, angle)
        right_mask = create_binary_mask(rotated_right)
        padded_left, padded_right = pad_to_same_size(left_mask.astype(float), right_mask.astype(float))
        score = calculate_overlap_score(padded_left > 0.5, padded_right > 0.5)
        scores[angle] = score
        if score > best_score:
            best_score, best_angle = score, angle
    return best_angle, best_score, scores

def detect_and_extract_images(left_crop, right_crop):
    if left_crop.dtype != object and len(left_crop.shape) == 2 and left_crop.shape[0] > 50 and left_crop.shape[1] > 50:
        return left_crop, right_crop
    
    if left_crop.dtype == object and len(left_crop.shape) == 2 and left_crop.shape[1] == 1:
        candidates = []
        for i in range(left_crop.shape[0]):
            try:
                candidate_left, candidate_right = left_crop[i, 0], right_crop[i, 0]
                if (hasattr(candidate_left, 'shape') and len(candidate_left.shape) == 2 and
                    hasattr(candidate_right, 'shape') and len(candidate_right.shape) == 2 and
                    candidate_left.shape[0] > 50 and candidate_left.shape[1] > 50):
                    candidates.append((i, candidate_left, candidate_right))
            except:
                continue
        
        if candidates:
            if MANUAL_INDEX_SELECTION is not None and 0 <= MANUAL_INDEX_SELECTION < len(candidates):
                return candidates[MANUAL_INDEX_SELECTION][1:3]
            return max(candidates, key=lambda x: np.max(x[1][x[1] > 0]) - np.min(x[1][x[1] > 0]))[1:3]
    
    return np.array(left_crop, dtype=float), np.array(right_crop, dtype=float)

def create_foot_overlay():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    mat = scipy.io.loadmat(MAT_FILE)
    scan_left, scan_right = detect_and_extract_images(mat["Indirect_plantar_Right_crop"], mat["Indirect_plantar_Left_crop"])
    
    if scan_left is None or scan_right is None:
        print("Failed to extract valid images.")
        return None
    
    img_left = trim_to_content(to_nan(scan_left))
    img_right_mir = mirror_horiz(trim_to_content(to_nan(scan_right)))
    
    rotation_angle, overlap_score, rotation_scores = 0, 0, {}
    if ENABLE_ROTATION_OPTIMIZATION:
        rotation_angle, overlap_score, rotation_scores = find_best_rotation_angle(img_left, img_right_mir)
        if rotation_angle != 0:
            img_right_mir = trim_to_content(rotate_image(img_right_mir, rotation_angle))
    
    scale_x = img_left.shape[1] / img_right_mir.shape[1] if img_right_mir.shape[1] else 1.0
    scale_y = img_left.shape[0] / img_right_mir.shape[0] if img_right_mir.shape[0] else 1.0
    img_right_scaled = scale_image(img_right_mir, scale_x, scale_y) if img_right_mir.shape != img_left.shape else img_right_mir

    if ENABLE_WARP_OPTIMIZATION:
        print("Starting improved warp optimization with temperature preservation...")
        
        # Store original temperature data for preservation
        original_temps = img_right_scaled[~np.isnan(img_right_scaled)]
        original_temp_mean = np.mean(original_temps)
        original_temp_std = np.std(original_temps)
        print(f"Original temperature stats: mean={original_temp_mean:.2f}°C, std={original_temp_std:.2f}°C")
        
        # Masks
        mask_left = create_binary_mask(img_left)
        mask_right = create_binary_mask(img_right_scaled)

        # To torch
        device = torch.device('cpu')
        mask_left_t = torch.from_numpy(mask_left).float().to(device)
        mask_right_t = torch.from_numpy(mask_right).float().to(device)

        # Create identity grid
        def create_grid(h, w):
            y = torch.linspace(-1, 1, h, device=device)
            x = torch.linspace(-1, 1, w, device=device)
            yy, xx = torch.meshgrid(y, x, indexing='ij')
            grid = torch.stack([xx, yy], dim=-1).unsqueeze(0)  # 1, h, w, 2
            return grid

        identity_grid = create_grid(*img_left.shape)

        # More conservative displacement parameters
        coarse_h, coarse_w = 6, 6  # Reduced from 4x4 to 3x3 for less aggressive warping
        displ_coarse = torch.nn.Parameter(torch.zeros(1, coarse_h, coarse_w, 2, device=device))

        # More conservative optimization parameters
        optimizer = torch.optim.Adam([displ_coarse], lr=0.02)  # Reduced learning rate from 0.05
        num_iters = 150  # Reduced iterations from 300

        best_iou = 0
        best_displacement = None

        for iter in range(num_iters):
            optimizer.zero_grad()

            # More conservative displacement scaling
            displ_full = torch.nn.functional.interpolate(
                displ_coarse.permute(0, 3, 1, 2), size=img_left.shape, mode='bilinear', align_corners=False
            ).permute(0, 2, 3, 1)

            grid = identity_grid + displ_full * 0.1  # Reduced from 0.2 to 0.1 for gentler warping

            warped = torch.nn.functional.grid_sample(
                mask_right_t.unsqueeze(0).unsqueeze(0), grid, mode='bilinear', padding_mode='zeros', align_corners=False
            ).squeeze()

            inter = torch.sum(warped * mask_left_t)
            union = torch.sum(warped + mask_left_t - warped * mask_left_t) + 1e-6
            iou = inter / union
            
            # Add regularization to prevent extreme deformations
            displacement_magnitude = torch.mean(torch.sum(displ_full ** 2, dim=-1))
            smoothness_loss = torch.mean(torch.abs(displ_full[:, 1:, :, :] - displ_full[:, :-1, :, :])) + \
                             torch.mean(torch.abs(displ_full[:, :, 1:, :] - displ_full[:, :, :-1, :]))
            
            loss = -iou + 0.1 * displacement_magnitude + 0.05 * smoothness_loss

            loss.backward()
            optimizer.step()
            
            # Track best solution
            if iou.item() > best_iou:
                best_iou = iou.item()
                best_displacement = displ_full.clone().detach()

            if iter % 30 == 0:  # Reduced logging frequency
                print(f"Iter {iter}: IOU {iou.item():.4f}, Displacement {displacement_magnitude.item():.4f}")

        # Use best displacement for final warping
        grid = identity_grid + best_displacement * 0.1

        # Apply warping to temperature data with better preservation
        img_right_t = torch.from_numpy(img_right_scaled).float().to(device)
        
        # Create a more conservative fill value based on actual data
        fill_value = float(np.nanmin(img_right_scaled)) - 5.0 if not np.isnan(np.nanmin(img_right_scaled)) else 15.0
        
        img_right_filled = torch.where(
            torch.isnan(img_right_t), torch.tensor(fill_value, dtype=torch.float32, device=device), img_right_t
        )
        valid_mask_t = (~torch.isnan(img_right_t)).float().to(device)

        # Use nearest neighbor to preserve exact temperature values
        warped_temp = torch.nn.functional.grid_sample(
            img_right_filled.unsqueeze(0).unsqueeze(0),
            grid,
            mode='nearest',  # Keep nearest neighbor for temperature preservation
            padding_mode='border',  # Use border padding instead of zeros
            align_corners=False
        ).squeeze()
        
        warped_valid = torch.nn.functional.grid_sample(
            valid_mask_t.unsqueeze(0).unsqueeze(0),
            grid,
            mode='nearest',
            padding_mode='zeros',
            align_corners=False
        ).squeeze()

        img_right_scaled = torch.where(
            warped_valid > 0.5, warped_temp, torch.tensor(np.nan, dtype=torch.float32, device=device)
        ).detach().cpu().numpy()

        # Validate temperature preservation
        final_temps = img_right_scaled[~np.isnan(img_right_scaled)]
        if len(final_temps) > 0:
            final_temp_mean = np.mean(final_temps)
            final_temp_std = np.std(final_temps)
            temp_drift = abs(final_temp_mean - original_temp_mean)
            print(f"Final temperature stats: mean={final_temp_mean:.2f}°C, std={final_temp_std:.2f}°C")
            print(f"Temperature drift: {temp_drift:.2f}°C")
            
            if temp_drift > 1.0:  # If temperature drifted too much, warn user
                print("⚠️  Warning: Significant temperature drift detected after warping")

        # Update overlap score
        overlap_score = calculate_overlap_score(create_binary_mask(img_left), create_binary_mask(img_right_scaled))
        print(f"Final overlap score: {overlap_score:.4f}")
        print("Warp optimization completed with improved temperature preservation.")
    left_canvas, right_canvas = pad_to_same_size(img_left, img_right_scaled)
    
    # Visualizations
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    for ax in axes[1, 1:]:
        ax.remove()
    
    for img, title, ax in [
        (img_left, 'Original Left Foot', axes[0, 0]),
        (img_right_mir, 'Right Foot (Mirrored)', axes[0, 1]),
        (img_right_scaled, 'Right Foot (Scaled & Mirrored)', axes[0, 2]),
        (left_canvas, 'Foot Overlay (alpha=0.6)', axes[1, 0])
    ]:
        im = ax.imshow(img, cmap=CMAP)
        ax.set_title(title)
        ax.axis('off')
        plt.colorbar(im, ax=ax)
    axes[1, 0].imshow(right_canvas, cmap=CMAP, alpha=0.6)
    
    output_path = os.path.join(OUTPUT_DIR, f"foot_overlay_comparison_{SELECTED_FILE}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    if ENABLE_FOCUSED_OVERLAY:
        fig2, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(left_canvas, cmap=CMAP)
        ax.imshow(right_canvas, cmap=CMAP, alpha=RIGHT_ALPHA)
        title = f'Foot Overlay Analysis\nScale: x={scale_x:.3f}, y={scale_y:.3f}' + (f', Rotation: {rotation_angle}°, Score: {overlap_score:.3f}' if ENABLE_ROTATION_OPTIMIZATION else '')
        ax.set_title(title)
        ax.axis('off')
        plt.colorbar(ax.imshow(left_canvas, cmap=CMAP), ax=ax, fraction=0.046, pad=0.04).set_label('Temperature (°C)')
        plt.savefig(os.path.join(OUTPUT_DIR, f"focused_foot_overlay_{SELECTED_FILE}.png"), dpi=300, bbox_inches='tight')
        plt.show()
    
    fig3, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 8))
    for ax, img, title in [
        (ax_left, left_canvas, 'Left Foot (Original)'),
        (ax_right, right_canvas, f'Right Foot (Mirrored, {"Rotated " + str(rotation_angle) + "°, " if rotation_angle else ""}Scaled)')
    ]:
        im = ax.imshow(img, cmap=CMAP)
        ax.set_title(title)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label('Temperature (°C)')
    
    fig_title = f'Side-by-Side Foot Comparison - {SELECTED_FILE.upper()}\nScale: x={scale_x:.3f}, y={scale_y:.3f}' + (f' | Rotation: {rotation_angle}° | Score: {overlap_score:.3f}' if ENABLE_ROTATION_OPTIMIZATION else '')
    fig3.suptitle(fig_title, fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(os.path.join(OUTPUT_DIR, f"sidebyside_feet_{SELECTED_FILE}.png"), dpi=300, bbox_inches='tight')
    plt.show()
    
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
    results = {}
    for file_key in AVAILABLE_MAT_FILES:
        global SELECTED_FILE, MAT_FILE
        SELECTED_FILE, MAT_FILE = file_key, AVAILABLE_MAT_FILES[file_key]
        try:
            result = create_foot_overlay()
            results[file_key] = "SUCCESS" if result is not None else "FAILED"
        except Exception as e:
            results[file_key] = f"ERROR: {str(e)}"
    print("\nSUMMARY OF ALL FILES:")
    for file_key, result in results.items():
        print(f"{file_key:10} : {result}")

if __name__ == "__main__":
    create_foot_overlay()
    # test_multiple_files()