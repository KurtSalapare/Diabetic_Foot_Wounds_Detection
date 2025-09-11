import os
import json
from scipy.io import loadmat
import numpy as np

def convert_mat_to_json(source_folder, output_folder):
    """
    Converts all .mat files in a source folder to .json files
    and saves them to an output folder.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    for filename in os.listdir(source_folder):
        if filename.endswith('.mat'):
            mat_path = os.path.join(source_folder, filename)
            
            try:
                # Load the .mat file
                mat_data = loadmat(mat_path)
                
                # Clean up the loaded data to make it JSON-serializable
                # This handles common SciPy data types like NumPy arrays.
                serializable_data = {}
                for key, value in mat_data.items():
                    # Skip metadata fields that start with '__'
                    if key.startswith('__'):
                        continue
                    
                    if isinstance(value, np.ndarray):
                        # Convert NumPy arrays to Python lists for JSON serialization
                        serializable_data[key] = value.tolist()
                    else:
                        serializable_data[key] = value
                
                # Create the output JSON file path
                json_filename = os.path.splitext(filename)[0] + '.json'
                json_path = os.path.join(output_folder, json_filename)
                
                # Save the data to a JSON file
                with open(json_path, 'w') as f:
                    json.dump(serializable_data, f, indent=4)
                    
                print(f"Successfully converted '{filename}' to '{json_filename}'")
                
            except Exception as e:
                print(f"Failed to convert '{filename}': {e}")
                
# Example Usage:
# Set the paths to your source folder containing .mat files
# and the destination folder for the JSON files.
source_directory = 'Diabetic_Foot_Wounds_Detection/Data/Temp Data/pnt_mat_files'
output_directory = 'Diabetic_Foot_Wounds_Detection/Data/Temp Data/json_format'

convert_mat_to_json(source_directory, output_directory)