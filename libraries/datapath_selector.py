import os
from pathlib import Path
import socket

def trim_before_escarabajos(path):
    if "escarabajos" in path:
        return path.split("escarabajos", 1)[0] + "escarabajos"
    else:
        return path  # Return the original path if "escarabajos" is not found
        
import os
import sys
from pathlib import Path

def get_paths():
    """
    Returns paths for collections, metadata, and other resources, using the parent directory
    of this script's location as the base folder.
    
    Returns:
        dict: Dictionary of paths to collections, metadata, and other resources.
    """
    # Determine the base folder as the parent of this script's directory
    script_dir = Path(__file__).resolve().parent  # C:\Users\esteb\escarabajos\libraries
    base_folder = script_dir.parent  # C:\Users\esteb\escarabajos
    
    # Add libraries folder to sys.path for module imports
    libraries_path = str(script_dir)
    if libraries_path not in sys.path:
        sys.path.append(libraries_path)
    
    # Define common paths relative to base_folder
    collection_tables_main_path = base_folder / "collections"
    collection_files_main_path = base_folder
    aggregated_data_location = base_folder / "aggregated_data"
    
    # Define paths dictionary
    paths = {
        '2018_2019_inbio_collection_path': collection_files_main_path / "CRAIC_data" / "corrected_files" / "2024-10-30" / "Mediciones Chrysina",
        '2018_2019_inbio_collection_metadata': collection_tables_main_path / "datos_especimenes_chrysina.txt",
        'angsol_collection_path': collection_files_main_path / "L1050_data" / "ANGSOL" / "average",
        'angsol_collection_metadata': collection_tables_main_path / "CICIMA-beetles-general-inventory - ANGSOL.txt",
        'cicimaucr_collection_path': collection_files_main_path / "L1050_data" / "TRA_data_CICIMA_INBUCR" / "CICIMAUCR" / "reflectance",
        'cicimaucr_collection_2_path': collection_files_main_path / "L1050_data" / "CICIMA-2024-01-REFLECTANCE" / "average",
        'cicimaucr_collection_3_path': collection_files_main_path / "L1050_data" / "CICIMA-2024-03-REFLECTANCE" / "without iris nor lens" / "average",
        'cicimaucr_collection_4_path': collection_files_main_path / "L1050_data" / "CICIMA-2024-05-REFLECTANCE" / "DORSAL" / "average",
        'cicima_ucr_metadata': collection_tables_main_path / "CICIMA-beetles-general-inventory - CICIMAUCR.txt",
        'inbucr_collection_path': collection_files_main_path / "L1050_data" / "INBUCR" / "average",
        'inbucr_metadata': collection_tables_main_path / "CICIMA-beetles-general-inventory - INBUCR.txt",
        'bioucr_collection_path': collection_files_main_path / "L1050_data" / "BIOUCR" / "average",
        'bioucr_metadata': collection_tables_main_path / "CICIMA-beetles-general-inventory - BIOUCR.txt",
        'aggregated_data_avg_path': aggregated_data_location / "peak_averages_krc.txt",
        'aggregated_data_std_path': aggregated_data_location / "peak_std_krc.txt",
        'aggregated_data_location': aggregated_data_location,
        'report_location': base_folder / "reports" / "data_analysis"
    }
    
    return paths

