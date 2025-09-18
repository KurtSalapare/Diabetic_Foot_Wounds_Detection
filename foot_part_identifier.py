import scipy.io
import os
import numpy as np
import matplotlib.pyplot as plt

from generate_heatspot_mask import trim_empty_columns

# ===========================
# Fixed Variables for Testing
# ===========================
day_index = 0
pnt = "Patient 1"
output_dir = "foot_images_folder"

# ===========================
# Helper Functions
# ===========================
def convert_to_2d_list(n_dimensional_array):
    
    # Check if the input is a NumPy array
    if not isinstance(n_dimensional_array, np.ndarray):
        raise TypeError("Input must be a NumPy ndarray.")
        
    # Reshape the array to a 2D structure if it's not already
    # This step is crucial if the input array is 1D or has more than 2 dimensions.
    # The -1 tells NumPy to infer the size of the first dimension.
    try:
        # Get the number of columns from the shape of the original array.
        # This assumes a 1D or 2D input. If the input is higher dimensional
        # e.g., (2, 3, 4), this will reshape it to (something, 12).
        # You may need to specify the exact number of columns if you want
        # to preserve the original structure.
        if n_dimensional_array.ndim > 2:
            print("Warning: Input has more than 2 dimensions. Reshaping into a flat 2D array.")
            reshaped_array = n_dimensional_array.reshape(-1, n_dimensional_array.shape[-1])
        else:
            reshaped_array = n_dimensional_array.reshape(n_dimensional_array.shape[0], -1)

    except ValueError as e:
        print(f"Error: Could not reshape the array. The number of elements may not be a multiple of the requested columns. Details: {e}")
        return ["error"]
        
    # 3. Convert the NumPy array to a Python list
    # The .tolist() method is the most efficient way to do this.
    converted_list = reshaped_array.tolist()
    converted_list = list(converted_list)
    
    return converted_list

def initial_vertical_split(two_dim_array):
    # Check if the input is a list and convert it to a NumPy array
    if isinstance(two_dim_array, list):
        two_dim_array = np.array(two_dim_array)
        
    # Ensure the input is a 2D NumPy array after the conversion attempt
    if not isinstance(two_dim_array, np.ndarray) or two_dim_array.ndim != 2:
        raise ValueError("Input must be a 2D NumPy array or a list of lists.")

    # Find the indices of all non-zero pixels (i.e., the foot)
    # This creates a mask of the foot, ignoring the zero-value background.
    non_zero_rows, zero_values = np.where(two_dim_array != 0)
    
    # Check if there are any non-zero pixels
    if non_zero_rows.size == 0:
        print("Warning: No non-zero values found in the image. Returning empty arrays.")
        return np.array([]), np.array([])
        
    # Find the bounding box of the non-zero values
    min_col = np.min(non_zero_rows)
    max_col = np.max(non_zero_rows)
    
    # Calculate the midpoint of the foot's width
    foot_midpoint = min_col + (max_col - min_col) // 2
    
    # Get the total number of rows and columns from the original image
    rows, cols = two_dim_array.shape
    print("Row Count : " + str(rows))
    print("Col Count : " + str(cols))
    
    # Use NumPy's slicing to create the two halves
    # Note: We still slice from the original image to maintain the
    # original dimensions for the background.
    # The left side is from the original start to the foot midpoint
    left_half = two_dim_array[:, :foot_midpoint]
    
    # The right side is from the foot midpoint to the end of the original image
    right_half = two_dim_array[:, foot_midpoint:]
    
    return left_half, right_half

def initial_horizontal_split(two_dim_array):
    # Check if the input is a list and convert it to a NumPy array
    if isinstance(two_dim_array, list):
        two_dim_array = np.array(two_dim_array)
        
    # Ensure the input is a 2D NumPy array after the conversion attempt
    if not isinstance(two_dim_array, np.ndarray) or two_dim_array.ndim != 2:
        raise ValueError("Input must be a 2D NumPy array or a list of lists.")
    
    non_zero_rows, zero_rows = np.where(two_dim_array != 0)
    
    print(str(len(non_zero_rows)))
    print(non_zero_rows)
    
    return [], []


# ---------------------------

# Directories #
## MAT-Files ##
mat_folder_path = "./Data/Temp Data"
if __name__ == "__main__":

    ## Testing With Healthy 1 ##
    mat_pnt_one = scipy.io.loadmat(mat_folder_path + "/gz1.mat")

    # Working on right foot for now #
    right_crop = mat_pnt_one["Direct_plantar_Right_crop"]
    scan_right = right_crop[day_index, 0]
    img_right = np.where(scan_right == 0, np.nan, scan_right)
    
    # print(img_right)
    print(len(img_right))
    print(len(img_right[0]))
    
    two_dim_array = np.array(img_right)
    if ((two_dim_array[0] != "error").any()):
        # print(two_dim_array)
        print(len(two_dim_array))
        print(len(two_dim_array[0]))
        # print(two_dim_array[100])
        
    # ==========================
    # Split image and then
    # Combine them to one with
    # A gap between them
    # ==========================
    
    left_side, right_side = initial_vertical_split(two_dim_array)
    print("Check if same : " + str(np.array_equal(left_side,right_side)))
    print(len(left_side))
    print(len(left_side[0]))
    print(len(right_side))
    print(len(right_side[0]))
    print(len(two_dim_array))
    
    top_left, bottom_left = initial_horizontal_split(left_side)
        
    # gap = np.full((left_side.shape[0], 5), np.nan)
    # combined = np.hstack((left_side, gap, right_side))

    # ==========================
    # Plot and save
    # ==========================

    # fig, ax = plt.subplots(figsize=(8, 6))
    # im = ax.imshow(combined, cmap="hot", interpolation='nearest')
    # ax.axis("off")
    # ax.set_title(f"Day {day_index+1}: Foot for {pnt} Left Side")

    # cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
    # cbar.set_label("Temperature (°C)")

    # plt.show()
    
    

    # plt.savefig(os.path.join(output_dir, f"foot_day{day_index+1}_left_side"),
    #             dpi=300, bbox_inches="tight")
    # plt.close()

    # os.makedirs(output_dir, exist_ok=True)
    # scipy.io.savemat(os.path.join(output_dir, f"{pnt}{day_index+1}_left_side.mat"),
    #                 {"foot_image": left_side})

    # print(f"Saved combined PNG in '{output_dir}' ({pnt})_left_side")