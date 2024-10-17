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
    collection_tables_main_path = base_folder / "collections"
    collection_files_main_path = base_folder 
    agregated_data_location = base_folder / "agregated_data"

    # Initialize paths dictionary
    #    paths are directories
    #    metadata are files
    paths = {
        '2018_2019_inbio_collection_path': collection_files_main_path/ "CRAIC_data" / "Mediciones Chrysina",
        '2018_2019_inbio_collection_metadata': collection_tables_main_path / "datos_especimenes_chrysina.txt",

        'angsol_collection_path': collection_files_main_path / "L1050_data"/ "ANGSOL" / "average",
        'angsol_collection_metadata': collection_tables_main_path / "CICIMA-beetles-general-inventory - ANGSOL.txt",
        
        'cicimaucr_collection_path': collection_files_main_path / "L1050_data"/ "TRA_data_CICIMA_INBUCR" / "CICIMAUCR" / "reflectance",
        'cicimaucr_collection_2_path': collection_files_main_path / "L1050_data"/ "CICIMA-2024-01-REFLECTANCE" / "average",
        'cicimaucr_collection_3_path': collection_files_main_path / "L1050_data"/ "CICIMA-2024-03-REFLECTANCE" / "without iris nor lens" / "average",
        'cicimaucr_collection_4_path': collection_files_main_path / "L1050_data"/ "CICIMA-2024-05-REFLECTANCE" / "DORSAL" / "average",
        'cicima_ucr_metadata': collection_tables_main_path / "CICIMA-beetles-general-inventory - CICIMAUCR.txt",

        'inbucr_collection_path': collection_files_main_path / "L1050_data"/ "INBUCR" / "average",
        'inbucr_metadata': collection_tables_main_path / "CICIMA-beetles-general-inventory - INBUCR.txt",

        'bioucr_collection_path': collection_files_main_path/ "L1050_data" / "BIOUCR" / "average",
        'bioucr_metadata': collection_tables_main_path / "CICIMA-beetles-general-inventory - BIOUCR.txt",

        'agregated_data_avg_path': agregated_data_location / "L1050_data"/ "peak_averages_krc.txt",
        'agregated_data_std_dev_path': agregated_data_location/ "L1050_data" / "peak_std_krc.txt",

        'report_location': base_folder / "reports" / "data_analysis"
    }

    return paths

