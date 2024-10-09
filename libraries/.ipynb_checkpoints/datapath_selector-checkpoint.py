import os

def get_paths(working_at):
    """OPTIONS: cicima_laptop, colaboratory, wfh, cicima_desktop
    """
    # Initialize a dictionary to hold the paths
    paths = {
        'angsol_collection_path': '',
        'angsol_collection_metadata': '',
        'craic_test_path': '',
        'cicimaucr_collection_path': '',
        'cicimaucr_collection_2_path': '',
        'cicimaucr_collection_3_path': '',
        'cicima_ucr_metadata': '',
        'inbucr_collection_path': '',
        'inbucr_metadata': '',
        'bioucr_collection_path': '',
        'bioucr_metadata': '',
        'agregated_data_avg_path': '',
        'agregated_data_std_dev_path': '',
        'report_location': '',
        'database_descriptor': ''
    }

    if working_at == "colaboratory":
        from google.colab import drive
        drive.mount("/content/drive")
        base_folder = r"/content/drive/My Drive/CICIMA/escarabajos_files/L1050_data/"
        
        paths['angsol_collection_path'] = base_folder + "ANGSOL//average"
        paths['angsol_collection_metadata'] = base_folder + r"collections/CICIMA-beetles-general-inventory - ANGSOL.txt"

        paths['cicimaucr_collection_path'] = base_folder + "CICIMA-2024-01-REFLECTANCE//average"
        paths['cicimaucr_collection_2_path'] = base_folder + "CICIMA-2024-03-REFLECTANCE//average"
        paths['cicimaucr_collection_3_path'] = base_folder + "TRA_data_CICIMA_INBUCR//CICIMAUCR//reflectance"
        paths['cicima_ucr_metadata'] = base_folder + r"collections/CICIMA-beetles-general-inventory - CICIMAUCR.txt"

        paths['inbucr_collection_path'] = base_folder + "INBUCR//average"
        paths['inbucr_metadata'] = base_folder + r"collections/CICIMA-beetles-general-inventory - INBUCR.txt"

        paths['bioucr_collection_path'] = base_folder + "BIOUCR//average"
        paths['bioucr_metadata'] = base_folder + r"collections/CICIMA-beetles-general-inventory - BIOUCR.txt"

        paths['agregated_data_avg_path'] = base_folder + "agregated_data/peak_averages_krc.txt"
        paths['agregated_data_std_dev_path'] = base_folder + "agregated_data/peak_std_krc.txt"

    elif working_at == "wfh":
        print("working from home")
        base_folder = r"C:\Users\EstebanSoto\Jupyter\escarabajos"
        collection_tables_main_path = os.path.join(base_folder, "L1050_data","collections")
        collection_files_main_path = os.path.join(base_folder, "L1050_data")
        report_location = r"C:\Users\EstebanSoto\Documents\Estudio Optico Escarabajos\data_analysis"
        
        paths['angsol_collection_path'] = os.path.join(collection_files_main_path, r"ANGSOL\average")
        paths['angsol_collection_metadata'] = os.path.join(collection_tables_main_path, "CICIMA-beetles-general-inventory - ANGSOL.txt")

        paths['cicimaucr_collection_path'] = os.path.join(collection_files_main_path, r"TRA_data_CICIMA_INBUCR\CICIMAUCR\reflectance")
        paths['cicimaucr_collection_2_path'] = os.path.join(collection_files_main_path, r"CICIMA-2024-01-REFLECTANCE\average")
        paths['cicimaucr_collection_3_path'] = os.path.join(collection_files_main_path, r"CICIMA-2024-03-REFLECTANCE\without iris nor lens\average")
        paths['cicima_ucr_metadata'] = os.path.join(collection_tables_main_path, r"CICIMA-beetles-general-inventory - CICIMAUCR.txt")

        paths['inbucr_collection_path'] = os.path.join(collection_files_main_path, r"INBUCR\average")
        paths['inbucr_metadata'] = os.path.join(collection_tables_main_path, r"CICIMA-beetles-general-inventory - INBUCR.txt")

        paths['bioucr_collection_path'] = os.path.join(collection_files_main_path, r"BIOUCR\average")
        paths['bioucr_metadata'] = os.path.join(collection_tables_main_path, r"CICIMA-beetles-general-inventory - BIOUCR.txt")

        agregated_data_location = os.path.join(base_folder, "agregated_data")
        paths['agregated_data_avg_path'] = os.path.join(agregated_data_location, "peak_averages_krc.txt")
        paths['agregated_data_std_dev_path'] = os.path.join(agregated_data_location, "peak_std_krc.txt")

    elif working_at == "cicima_desktop":
        base_folder = r"C:\Users\esteb\cicima\escarabajos"
        collection_tables_main_path = os.path.join(base_folder, "L1050_data","collections")
        collection_files_main_path = os.path.join(base_folder, "L1050_data")
        report_location = r"C:\Users\EstebanSoto\Documents\Estudio Optico Escarabajos\data_analysis"
        database_descriptor = r"CICIMAUCR and ANGSOL"

        paths['angsol_collection_path'] = os.path.join(collection_files_main_path, r"ANGSOL","average")
        paths['angsol_collection_metadata'] = os.path.join(collection_tables_main_path, "CICIMA-beetles-general-inventory - ANGSOL.txt")

        paths['cicimaucr_collection_path'] = os.path.join(collection_files_main_path, r"TRA_data_CICIMA_INBUCR\CICIMAUCR\reflectance")
        paths['cicimaucr_collection_2_path'] = os.path.join(collection_files_main_path, r"CICIMA-2024-01-REFLECTANCE\average")
        paths['cicimaucr_collection_3_path'] = os.path.join(collection_files_main_path, r"CICIMA-2024-03-REFLECTANCE\without iris nor lens\average")
        paths['cicima_ucr_metadata'] = os.path.join(collection_tables_main_path, r"CICIMA-beetles-general-inventory - CICIMAUCR.txt")

        paths['inbucr_collection_path'] = os.path.join(collection_files_main_path, r"INBUCR","average")
        paths['inbucr_metadata'] = os.path.join(collection_tables_main_path, r"CICIMA-beetles-general-inventory - INBUCR.txt")

        paths['bioucr_collection_path'] = os.path.join(collection_files_main_path, r"BIOUCR","average")
        paths['bioucr_metadata'] = os.path.join(collection_tables_main_path, r"CICIMA-beetles-general-inventory - BIOUCR.txt")

        agregated_data_location = os.path.join(base_folder, "agregated_data")
        paths['agregated_data_avg_path'] = os.path.join(agregated_data_location, "peak_averages_krc.txt")
        paths['agregated_data_std_dev_path'] = os.path.join(agregated_data_location, "peak_std_krc.txt")

    elif working_at == "cicima_laptop":
        base_folder = r"/home/vinicio/escarabajos"
        collection_tables_main_path = os.path.join(base_folder, "L1050_data","collections")
        collection_files_main_path = os.path.join(base_folder, "L1050_data")
        report_location = os.path.join(base_folder, "reports","data_analysis")
        database_descriptor = r"CICIMAUCR and ANGSOL"

        paths['angsol_collection_path'] = os.path.join(collection_files_main_path, r"ANGSOL","average")
        paths['angsol_collection_metadata'] = os.path.join(collection_tables_main_path, "CICIMA-beetles-general-inventory - ANGSOL.txt")

        paths['cicimaucr_collection_path'] = os.path.join(collection_files_main_path, r"TRA_data_CICIMA_INBUCR","CICIMAUCR","reflectance")
        paths['cicimaucr_collection_2_path'] = os.path.join(collection_files_main_path, r"CICIMA-2024-01-REFLECTANCE","average")
        paths['cicimaucr_collection_3_path'] = os.path.join(collection_files_main_path, r"CICIMA-2024-03-REFLECTANCE","without iris nor lens","average")
        paths['cicima_ucr_metadata'] = os.path.join(collection_tables_main_path, r"CICIMA-beetles-general-inventory - CICIMAUCR.txt")

        paths['inbucr_collection_path'] = os.path.join(collection_files_main_path, r"INBUCR","average")
        paths['inbucr_metadata'] = os.path.join(collection_tables_main_path, r"CICIMA-beetles-general-inventory - INBUCR.txt")

        paths['bioucr_collection_path'] = os.path.join(collection_files_main_path, r"BIOUCR","average")
        paths['bioucr_metadata'] = os.path.join(collection_tables_main_path, r"CICIMA-beetles-general-inventory - BIOUCR.txt")

        agregated_data_location = os.path.join(base_folder, "agregated_data")
        paths['agregated_data_avg_path'] = os.path.join(agregated_data_location, "peak_averages_krc.txt")
        paths['agregated_data_std_dev_path'] = os.path.join(agregated_data_location, "peak_std_krc.txt")
    elif working_at != "wfh":
        raise Exception("Invalid entry for working_at")
    return paths
