import os
from pathlib import Path
import socket

def get_paths():
    """Selects paths based on the system hostname."""
    # Map hostnames to base folders
    hostname = socket.gethostname()
    base_folders = {
        "colaboratory": "/content/drive/My Drive/CICIMA/escarabajos_files/L1050_data",
        "Shannon": r"C:\Users\esteb\cicima\escarabajos",
        "CICIMA-EVSM": r"C:\Users\EstebanSoto\Jupyter\escarabajos",
        "cicima_laptop": "/home/vinicio/escarabajos"
    }

    # Validate if the hostname is recognized
    if hostname not in base_folders:
        raise ValueError(f"Unknown hostname: {hostname}")

    base_folder = Path(base_folders[hostname])

    # Define common paths
    collection_tables_main_path = base_folder / "L1050_data" / "collections"
    collection_files_main_path = base_folder / "L1050_data"
    agregated_data_location = base_folder / "agregated_data"

    # Initialize paths dictionary
    paths = {
        'angsol_collection_path': collection_files_main_path / "ANGSOL" / "average",
        'angsol_collection_metadata': collection_tables_main_path / "CICIMA-beetles-general-inventory - ANGSOL.txt",
        
        'cicimaucr_collection_path': collection_files_main_path / "TRA_data_CICIMA_INBUCR" / "CICIMAUCR" / "reflectance",
        'cicimaucr_collection_2_path': collection_files_main_path / "CICIMA-2024-01-REFLECTANCE" / "average",
        'cicimaucr_collection_3_path': collection_files_main_path / "CICIMA-2024-03-REFLECTANCE" / "without iris nor lens" / "average",
        'cicimaucr_collection_4_path': collection_files_main_path / "CICIMA-2024-05-REFLECTANCE" / "DORSAL" / "average",
        'cicima_ucr_metadata': collection_tables_main_path / "CICIMA-beetles-general-inventory - CICIMAUCR.txt",

        'inbucr_collection_path': collection_files_main_path / "INBUCR" / "average",
        'inbucr_metadata': collection_tables_main_path / "CICIMA-beetles-general-inventory - INBUCR.txt",

        'bioucr_collection_path': collection_files_main_path / "BIOUCR" / "average",
        'bioucr_metadata': collection_tables_main_path / "CICIMA-beetles-general-inventory - BIOUCR.txt",

        'agregated_data_avg_path': agregated_data_location / "peak_averages_krc.txt",
        'agregated_data_std_dev_path': agregated_data_location / "peak_std_krc.txt",

        'report_location': base_folder / "reports" / "data_analysis"
    }

    return paths

