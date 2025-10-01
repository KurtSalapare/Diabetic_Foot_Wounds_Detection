import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.ndimage import zoom, rotate, map_coordinates, median_filter
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
    "pnt3": "Data/Temp Data/pnt3.mat",
}
SELECTED_FILE = "gz7"
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

# ---------- Utility functions (mostly unchanged) ----------

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

# ---------- New: affine pre-registration via centroid + PCA orientation ----------
def estimate_centroid_and_orientation(mask):
    """Return centroid (y, x) and principal axis angle (radians) for binary mask."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return (mask.shape[0] / 2.0, mask.shape[1] / 2.0), 0.0
    centroid = (np.mean(ys), np.mean(xs))
    # PCA on coordinates to get orientation
    coords = np.vstack([xs - centroid[1], ys - centroid[0]]).T  # Nx2 (x,y)
    cov = np.cov(coords, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # largest eigenvector is principal axis (x-direction)
    principal = eigvecs[:, np.argmax(eigvals)]
    angle = np.arctan2(principal[1], principal[0])  # radians (y over x)
    return centroid, angle

def apply_affine_transform_numpy(img, M, output_shape, fill_value=np.nan):
    """Apply 2x3 affine M (in pixel coordinates) to img using nearest neighbor and preserving NaNs.
       M maps output coords to input coords if we use np.float32 grid. We'll invert M if needed.
    """
    # Build coordinate grid for output image and compute source coords
    H, W = output_shape
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    coords = np.vstack([xs.ravel(), ys.ravel(), np.ones_like(xs.ravel())])  # 3 x (H*W)
    # Need to use inverse mapping: for each output pixel, find the source pixel that maps to it.
    # If M is 2x3 mapping src -> dst, we invert it. We'll accept M as src->dst and invert.
    A = np.vstack([M, [0,0,1]])  # 3x3
    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        A_inv = np.linalg.pinv(A)
    src = A_inv.dot(coords)  # 3 x (H*W)
    src_x = src[0, :].reshape(H, W)
    src_y = src[1, :].reshape(H, W)
    # nearest neighbor sampling
    src_x_nn = np.rint(src_x).astype(int)
    src_y_nn = np.rint(src_y).astype(int)
    sampled = np.full((H, W), np.nan, dtype=float)
    valid = (src_x_nn >= 0) & (src_x_nn < img.shape[1]) & (src_y_nn >= 0) & (src_y_nn < img.shape[0])
    sampled[valid] = img[src_y_nn[valid], src_x_nn[valid]]
    return sampled

def affine_pre_registration(right_img, left_img):
    """Estimate a simple affine (rotation + isotropic scale + translation) to roughly align right_img to left_img."""
    # Create masks
    mask_left = create_binary_mask(left_img)
    mask_right = create_binary_mask(right_img)
    if mask_left.sum() == 0 or mask_right.sum() == 0:
        # nothing to do
        return right_img, np.eye(3)[:2,:]

    # Centroids and orientation
    centroid_left, angle_left = estimate_centroid_and_orientation(mask_left)
    centroid_right, angle_right = estimate_centroid_and_orientation(mask_right)
    angle_diff = angle_left - angle_right  # rotate right to match left

    # Compute isotropic scale from bounding box sizes
    ys_l, xs_l = np.where(mask_left)
    ys_r, xs_r = np.where(mask_right)
    bbox_h_l = ys_l.max() - ys_l.min() + 1
    bbox_w_l = xs_l.max() - xs_l.min() + 1
    bbox_h_r = ys_r.max() - ys_r.min() + 1
    bbox_w_r = xs_r.max() - xs_r.min() + 1
    # use mean scale of height and width
    scale_h = bbox_h_l / max(bbox_h_r, 1)
    scale_w = bbox_w_l / max(bbox_w_r, 1)
    scale = (scale_h + scale_w) / 2.0
    # clamp scale to reasonable range to avoid blowing things up
    scale = float(np.clip(scale, 0.6, 1.6))

    # Build affine matrix (src -> dst) in pixel coordinates:
    # Steps: scale -> rotate -> translate to align centroids
    # We'll form M such that [x_dst, y_dst, 1]^T = A * [x_src, y_src, 1]^T
    cos_t = np.cos(angle_diff)
    sin_t = np.sin(angle_diff)
    R = np.array([[cos_t, -sin_t],
                  [sin_t,  cos_t]])
    S = np.eye(2) * scale
    RS = R.dot(S)
    # translation: move right centroid (after RS and rotation around origin) to left centroid
    # compute where right centroid maps (without translation)
    rc = np.array([centroid_right[1], centroid_right[0]])  # (x,y)
    mapped_rc = RS.dot(rc)
    lc = np.array([centroid_left[1], centroid_left[0]])
    t = lc - mapped_rc
    M = np.zeros((2,3), dtype=float)
    M[:, :2] = RS
    M[:, 2] = t
    # Apply to right_img canvas shaped like left_img for easier downstream processing
    H_out, W_out = left_img.shape
    # Use nearest neighbor to preserve temps (and preserve NaNs)
    right_affined = apply_affine_transform_numpy(right_img, M, (H_out, W_out), fill_value=np.nan)
    return right_affined, M

# ---------- New: Multi-scale warping pipeline (coarse-to-fine) ----------
def multi_scale_warp_preserve_temperatures(img_src, mask_src, img_ref, mask_ref,
                                           scales=[4, 8],  # list of coarse-to-fine control grid sizes
                                           iterations_per_scale=[120, 90],
                                           lr_per_scale=[0.04, 0.02],
                                           displacement_scale_factor=0.12,
                                           device=torch.device('cpu')):
    """
    img_src: numpy array (H_ref, W_ref) - source image to warp (already roughly affine-aligned & same canvas as img_ref)
    mask_src: binary mask for source (bool)
    img_ref: numpy array (H_ref, W_ref) - target image
    mask_ref: binary mask for target (bool)
    Returns warped_img (numpy) and final_iou
    """
    H, W = img_ref.shape
    # Prepare torch tensors
    img_src_t = torch.from_numpy(np.nan_to_num(img_src, nan=0.0)).float().to(device)
    valid_mask_src_t = torch.from_numpy((~np.isnan(img_src)).astype(np.float32)).float().to(device)
    mask_ref_t = torch.from_numpy(mask_ref.astype(np.float32)).float().to(device)
    identity_grid = lambda h,w: (lambda yy,xx: torch.stack([xx, yy], dim=-1).unsqueeze(0))(*torch.meshgrid(torch.linspace(-1,1,h,device=device), torch.linspace(-1,1,w,device=device), indexing='ij'))

    best_overall_iou = 0.0
    best_overall_displacement = None

    # progressive optimization across scales
    for s_idx, grid_size in enumerate(scales):
        iters = iterations_per_scale[min(s_idx, len(iterations_per_scale)-1)]
        lr = lr_per_scale[min(s_idx, len(lr_per_scale)-1)]
        # coarse displacement grid has shape 1 x gh x gw x 2
        gh, gw = grid_size, grid_size
        displ_coarse = torch.nn.Parameter(torch.zeros(1, gh, gw, 2, device=device))
        optimizer = torch.optim.Adam([displ_coarse], lr=lr)

        for it in range(iters):
            optimizer.zero_grad()
            # upsample to full resolution
            displ_full = torch.nn.functional.interpolate(
                displ_coarse.permute(0, 3, 1, 2), size=(H, W), mode='bilinear', align_corners=False
            ).permute(0, 2, 3, 1)  # 1,H,W,2

            # scale displacement magnitude (smaller factor to be conservative)
            grid = identity_grid(H,W) + displ_full * displacement_scale_factor

            # warp the source mask (use bilinear for mask's soft measurement)
            warped_mask = torch.nn.functional.grid_sample(
                valid_mask_src_t.unsqueeze(0).unsqueeze(0), grid, mode='bilinear', padding_mode='zeros', align_corners=False
            ).squeeze()
            # binarize softly
            warped_mask_bin = (warped_mask > 0.5).float()

            inter = torch.sum(warped_mask_bin * mask_ref_t)
            union = torch.sum(warped_mask_bin + mask_ref_t - warped_mask_bin * mask_ref_t) + 1e-6
            iou = inter / union

            # regularization terms
            displacement_magnitude = torch.mean(torch.sum(displ_full ** 2, dim=-1))
            smoothness_loss = torch.mean(torch.abs(displ_full[:, 1:, :, :] - displ_full[:, :-1, :, :])) + \
                              torch.mean(torch.abs(displ_full[:, :, 1:, :] - displ_full[:, :, :-1, :]))

            # combine into loss (maximize IoU)
            loss = -iou + 0.08 * displacement_magnitude + 0.04 * smoothness_loss

            loss.backward()
            optimizer.step()

            if iou.item() > best_overall_iou:
                best_overall_iou = iou.item()
                best_overall_displacement = displ_full.clone().detach()

            # occasional logging
            if (it % max(1, iters//5)) == 0:
                # print progress
                print(f"[Scale {grid_size}] Iter {it}/{iters} - IOU: {iou.item():.4f}, disp_mag: {displacement_magnitude.item():.4f}")

    # After all scales, use best_overall_displacement to warp the actual temperature image with nearest neighbor
    if best_overall_displacement is None:
        # nothing improved
        final_iou = calculate_overlap_score(mask_ref, mask_src)
        return img_src, final_iou

    final_grid = identity_grid(H, W) + best_overall_displacement * displacement_scale_factor

    # Convert grid from [-1,1] to grid_sample inputs and warp temperature image
    # Prepare padded source image (fill NaNs with a conservative fill based on min value)
    fill_value = float(np.nanmin(img_src)) - 5.0 if not np.isnan(np.nanmin(img_src)) else 15.0
    img_src_filled_t = torch.where(torch.isnan(torch.from_numpy(img_src).float().to(device)),
                                   torch.tensor(fill_value, dtype=torch.float32, device=device),
                                   torch.from_numpy(img_src).float().to(device))
    img_src_filled_t = img_src_filled_t.unsqueeze(0).unsqueeze(0)

    warped_temp_t = torch.nn.functional.grid_sample(
        img_src_filled_t, final_grid, mode='nearest', padding_mode='border', align_corners=False
    ).squeeze()

    warped_valid_t = torch.nn.functional.grid_sample(
        torch.from_numpy((~np.isnan(img_src)).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device),
        final_grid, mode='nearest', padding_mode='zeros', align_corners=False
    ).squeeze()

    warped_np = warped_temp_t.detach().cpu().numpy()
    warped_valid_np = warped_valid_t.detach().cpu().numpy()

    # Force NaNs where warped_valid is 0
    warped_np_final = np.where(warped_valid_np > 0.5, warped_np, np.nan)

    # Small post-process median filter to remove single-pixel jaggies while preserving actual temperature values.
    # We apply median filter only on valid pixels region (we temporarily fill NaNs with a sentinel, median-filter,
    # and then restore NaNs outside valid region). Because we used nearest sampling, median won't invent new temps far from
    # the true values; it's just removing local isolated blocks.
    if np.any(~np.isnan(warped_np_final)):
        sentinel = fill_value - 1000.0  # far outside valid temp range
        temp_copy = np.where(np.isnan(warped_np_final), sentinel, warped_np_final)
        # median filter size small (3x3)
        temp_med = median_filter(temp_copy, size=3, mode='nearest')
        # restore: keep median only where original pixel was valid (to avoid smoothing across NaN boundaries),
        # and where the median is not the sentinel.
        smoothed = np.where(temp_copy != sentinel, temp_med, np.nan)
        # For safety, only replace pixels where smoothed differs slightly (avoid changing real temperatures drastically)
        diff = np.abs(np.where(np.isnan(warped_np_final), 0, warped_np_final) - np.where(np.isnan(smoothed), 0, smoothed))
        # Allow replacements up to a small threshold (0.5°C). You can increase if you want stronger smoothing.
        replace_mask = (~np.isnan(smoothed)) & ((diff <= 0.5) | np.isnan(warped_np_final))
        warped_np_final[replace_mask] = smoothed[replace_mask]

    final_iou = calculate_overlap_score(mask_ref, ~np.isnan(warped_np_final))
    return warped_np_final, final_iou

# ---------- Main overlay creation (updated pipeline) ----------
def create_foot_overlay():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    mat = scipy.io.loadmat(MAT_FILE)
    scan_left, scan_right = detect_and_extract_images(mat["Dorsal_Right_crop"], mat["Dorsal_Left_crop"])
    
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
    
    # Pre-registration (new): affine alignment (centroid + PCA-based orientation + isotropic scale)
    img_right_affined, M_affine = affine_pre_registration(img_right_mir, img_left)
    # After affining, we may need to trim to content again
    img_right_affined = trim_to_content(img_right_affined)
    
    # Compute scale based on shapes (but since affine already resized/cropped onto left canvas, we'll re-evaluate)
    # If shapes differ, scale image_right_affined to img_left shape using nearest sampling to preserve temps
    if img_right_affined.shape != img_left.shape:
        # nearest-neighbor resampling to exactly match left canvas size
        H_l, W_l = img_left.shape
        H_r, W_r = img_right_affined.shape
        # compute new coordinates mapping
        ys_new, xs_new = np.mgrid[0:H_l, 0:W_l]
        src_x = (xs_new * (W_r / W_l)).astype(int)
        src_y = (ys_new * (H_r / H_l)).astype(int)
        src_x = np.clip(src_x, 0, W_r - 1)
        src_y = np.clip(src_y, 0, H_r - 1)
        img_right_scaled = img_right_affined[src_y, src_x]
    else:
        img_right_scaled = img_right_affined

    # If not performing optimization, fallback: simple scale pad
    if not ENABLE_WARP_OPTIMIZATION:
        left_canvas, right_canvas = pad_to_same_size(img_left, img_right_scaled)
        # Visualize and exit
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(left_canvas, cmap=CMAP); axes[0].set_title("Left")
        axes[1].imshow(right_canvas, cmap=CMAP); axes[1].set_title("Right (affined)")
        plt.show()
        return {
            'left_original': img_left,
            'right_mirrored': img_right_mir,
            'right_scaled': img_right_scaled,
            'left_canvas': left_canvas,
            'right_canvas': right_canvas,
            'scale_x': 1.0,
            'scale_y': 1.0,
            'rotation_angle': rotation_angle,
            'overlap_score': overlap_score,
            'rotation_scores': rotation_scores
        }

    # Warp optimization (multi-scale)
    print("Starting multi-scale warp optimization with temperature preservation...")

    # Masks (after affined + scaling)
    mask_left = create_binary_mask(img_left)
    mask_right = create_binary_mask(img_right_scaled)

    # Ensure same canvas sizes
    if img_right_scaled.shape != img_left.shape:
        img_right_scaled, img_left = pad_to_same_size(img_right_scaled, img_left)[0], pad_to_same_size(img_right_scaled, img_left)[1]
        # recompute masks
        mask_left = create_binary_mask(img_left)
        mask_right = create_binary_mask(img_right_scaled)

    # Call multi-scale optimizer
    # choose scales: begin coarse (4x4), then 8x8 (or 6->12 depending on needs)
    scales = [4, 8]
    iterations = [140, 120]  # modest iters per scale
    lrs = [0.05, 0.02]
    warped_right, final_iou = multi_scale_warp_preserve_temperatures(
        img_right_scaled, mask_right, img_left, mask_left,
        scales=scales, iterations_per_scale=iterations, lr_per_scale=lrs,
        displacement_scale_factor=0.12,
        device=torch.device('cpu')
    )

    print(f"Completed warp optimization. Final IOU: {final_iou:.4f}")

    left_canvas, right_canvas = pad_to_same_size(img_left, warped_right)

    # Visualizations (same as before, with updated canvases)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    for ax in axes[1, 1:]:
        ax.remove()
    
    for img, title, ax in [
        (img_left, 'Original Left Foot', axes[0, 0]),
        (img_right_mir, 'Right Foot (Mirrored)', axes[0, 1]),
        (img_right_scaled, 'Right Foot (Affined -> Scaled)', axes[0, 2]),
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
        title = f'Foot Overlay Analysis (Affine + Multi-scale Warp)\n' + (f'Rotation: {rotation_angle}°, Score: {final_iou:.3f}' if ENABLE_ROTATION_OPTIMIZATION else '')
        ax.set_title(title)
        ax.axis('off')
        plt.colorbar(ax.imshow(left_canvas, cmap=CMAP), ax=ax, fraction=0.046, pad=0.04).set_label('Temperature (°C)')
        plt.savefig(os.path.join(OUTPUT_DIR, f"focused_foot_overlay_{SELECTED_FILE}.png"), dpi=300, bbox_inches='tight')
        plt.show()
    
    fig3, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 8))
    for ax, img, title in [
        (ax_left, left_canvas, 'Left Foot (Original)'),
        (ax_right, right_canvas, f'Right Foot (Affined & Warped) - IOU: {final_iou:.3f}')
    ]:
        im = ax.imshow(img, cmap=CMAP)
        ax.set_title(title)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label('Temperature (°C)')
    
    fig_title = f'Side-by-Side Foot Comparison - {SELECTED_FILE.upper()}\n' + (f' | Rotation: {rotation_angle}° | Final IOU: {final_iou:.3f}' if ENABLE_ROTATION_OPTIMIZATION else '')
    fig3.suptitle(fig_title, fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(os.path.join(OUTPUT_DIR, f"sidebyside_feet_{SELECTED_FILE}.png"), dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'left_original': img_left,
        'right_mirrored': img_right_mir,
        'right_affined': img_right_affined,
        'right_scaled': img_right_scaled,
        'right_warped': warped_right,
        'left_canvas': left_canvas,
        'right_canvas': right_canvas,
        'rotation_angle': rotation_angle,
        'overlap_score': final_iou,
        'rotation_scores': rotation_scores,
        'affine_matrix': M_affine
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
