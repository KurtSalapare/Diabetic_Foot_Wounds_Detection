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
    foot_temperatures = data.get("Dorsal_Left_crop")
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

# Example usage:
# First, you would need to have your temperature data in a JSON file.
# I'll create a dummy one for demonstration.
json_path = "./Diabetic_Foot_Wounds_Detection/Data/Temp Data/json_format/pnt1.json"


# Testing_Image_Generation_from_Json
generate_thermal_image_from_json(json_path, "foot_thermogram_test_2.png")
