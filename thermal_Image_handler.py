import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def generate_thermal_image_from_json(json_file_path, image_name="thermal_image.png"):
    """
    Generates a thermal image from a JSON file containing a 2D array of
    temperature values.

    Args:
        json_file_path (str): The file path to the JSON file.
        image_name (str): The desired name for the output image file.
    """
    try:
        # Load the JSON data
        with open(json_file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file at '{json_file_path}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{json_file_path}'.")
        return

    # Assuming the data for the left foot is under this key
    foot_temperatures = data.get("Direct_plantar_Left_crop")
    if foot_temperatures is None:
        print("Error: 'Direct_plantar_Left' key not found in the JSON file.")
        return

    # Convert the list of lists to a NumPy array for easier plotting
    temp_array = np.array(foot_temperatures)

    # Create a figure and axes for the plot
    fig, ax = plt.subplots(figsize=(temp_array.shape[1] / 100, temp_array.shape[0] / 100))

    # Display the data as a thermal image using a colormap
    # 'hot' is a good colormap for thermal data, but you can also use 'inferno', 'plasma', etc.
    # The `interpolation='bilinear'` smoothens the pixelated image.
    c = ax.imshow(temp_array, cmap='hot', interpolation='bilinear')

    # Add a color bar to show the temperature scale
    cbar = fig.colorbar(c, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Temperature (°C)', rotation=270, labelpad=15)

    # Set the title and labels
    ax.set_title('Thermal Image of the Foot')
    ax.set_xticks([])
    ax.set_yticks([])

    # Save the figure as a high-quality image file
    plt.savefig(image_name, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()  # Close the figure to free up memory

    print(f"Thermal image saved successfully as '{image_name}'")

def change_values_til_row(json_data, key, row_index):
    """
    Changes all temperature values to 0 for all rows up to a specific row index.

    Args:
        json_data (dict): The dictionary containing the temperature data.
        key (str): The key for the 2D array in the dictionary (e.g., "Direct_plantar_Left").
        row_index (int): The index of the row to zero out, inclusive.

    Returns:
        dict: The modified dictionary.
    """
    if key in json_data and isinstance(json_data[key], list):
        if 0 < len(json_data[key]):
            # Iterate through all rows up to the specified row_index
            for i in range(len(json_data[key])):
                j = 0
                while j < len(json_data[key][i]):
                    # print("Before: " + str(json_data[key][i][j]))
                    json_data[key][i][j] = json_data[key][i][j] - row_index 
                    # print("After: " + str(json_data[key][i][j]))
                    j+=1
        else:
            print(f"Error: Row index {row_index} is out of bounds.")
    else:
        print(f"Error: Key '{key}' not found or is not a list.")
        
    print(f"All values increased by {row_index}")
    return json_data

def zero_row_value_checker(json_data, key, row_index):
    """
    Changes all temperature values to 0 for all rows up to a specific row index.

    Args:
        json_data (dict): The dictionary containing the temperature data.
        key (str): The key for the 2D array in the dictionary (e.g., "Direct_plantar_Left").
        row_index (int): The index of the row to zero out, inclusive.

    Returns:
        check (bool): Checks if the values are infact 0
    """
    
    check = True
    if key in json_data and isinstance(json_data[key], list):
        if 0 <= row_index < len(json_data[key]):
            # Iterate through all rows up to the specified row_index
            for i in range(row_index + 1):
                j = 0
                while j < len(json_data[key][i]):
                    if json_data[key][i][j] != 0:
                        check = False
                        return check
                    j+=1
                    
        else:
            print(f"Error: Row index {row_index} is out of bounds.")
    else:
        print(f"Error: Key '{key}' not found or is not a list.")
        
    print(f"All rows from 0 to {row_index} in '{key}' have been checked if they are zeroed out.")
    return check

def generate_thermal_image(json_dict, image_name="thermal_image.png"):
    """
    Generates a thermal image from a JSON file containing a 2D array of
    temperature values.

    Args:
        json_dict (dict): The file path to the JSON file.
        image_name (str): The desired name for the output image file.
    """

    # Assuming the data for the left foot is under this key
    foot_temperatures = json_dict.get("Direct_plantar_Left_crop")
    print(foot_temperatures)
    if foot_temperatures is None:
        print("Error: 'Direct_plantar_Left_crop' key not found in the JSON file.")
        return

    # Convert the list of lists to a NumPy array for easier plotting
    temp_array = np.array(foot_temperatures)

    # Create a figure and axes for the plot
    fig, ax = plt.subplots(figsize=(temp_array.shape[1] / 100, temp_array.shape[0] / 100))

    # Display the data as a thermal image using a colormap
    # 'hot' is a good colormap for thermal data, but you can also use 'inferno', 'plasma', etc.
    # The `interpolation='bilinear'` smoothens the pixelated image.
    c = ax.imshow(temp_array, cmap='hot', interpolation='bilinear')

    # Add a color bar to show the temperature scale
    cbar = fig.colorbar(c, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Temperature (°C)', rotation=270, labelpad=15)

    # Set the title and labels
    ax.set_title('Thermal Image of the Foot')
    ax.set_xticks([])
    ax.set_yticks([])

    # Save the figure as a high-quality image file
    plt.savefig(image_name, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()  # Close the figure to free up memory

    print(f"Thermal image saved successfully as '{image_name}'")


# Example usage:
# First, you would need to have your temperature data in a JSON file.
# I'll create a dummy one for demonstration.
json_path = "./Diabetic_Foot_Wounds_Detection/Data/Temp Data/json_format/pnt1.json"

# Testing_Image_Generation_from_Json
# generate_thermal_image_from_json(json_path, "foot_thermogram_test.png")


# Testing_Image_Generation_from_Json
try:
    with open(json_path, 'r') as f:
        foot_data = json.load(f)
except FileNotFoundError:
    print(f"Error: The file at '{json_path}' was not found. Please ensure the path is correct.")
    foot_data = None
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from '{json_path}'.")
    foot_data = None
    
if foot_data:
    # First, zero out a row (e.g., row 10)
    zeroed_data = change_values_til_row(foot_data, "Direct_plantar_Left_crop", 10)
    
    
    # print(zero_row_value_checker(zeroed_data, "Direct_plantar_Left_crop", 20))
    # Then, generate a thermal image from the modified data
    generate_thermal_image(zeroed_data, "modified_foot_thermogram.png")

