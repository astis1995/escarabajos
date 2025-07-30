print("Getting imports")
use_cuda = True
print("Using CUDA: ", use_cuda)
 
import os
#from image_utilities import group_by_v_h
import image_utilities

if use_cuda: 
    from stitching_cuda import stitch_images, get_overlaps_static
else:
    from stitching import stitch_images

from volume_reconstruction import load_panorama_slices, optimized_marching_cubes_to_trimesh
from poisson_reconstruction_2 import *
#from preprocessing import get_image_paths, preprocessing_routine, average_and_median_images
import preprocessing
import numpy as np
from pathlib import Path
from overlap_utils import get_optimal_overlap
import time
import traceback
import gc 

BASE_DIR = os.getcwd()
Y_CUT_DISTANCE_MM = 1.0


#configurations
IS_SINGLE_SCAN = False

#stages

AVERAGES_AND_MEDIAN_DONE = True
PREPROCESSING_DONE = False
STITCHING_DONE = False
MARCHING_CUBES_DONE = False
POISSON_RECONSTRUCTION_DONE = False


#

def save_stats(
    averaging_and_median_time: float,
    preprocessing_time: float,
    stitching_time: float,
    marching_cubes_time: float,
    poisson_reconstruction_time: float,
    elapsed_time: float,
    valid_files: int,
    output_folder: str = "timing_logs",
    csv_filename: str = "timing_log.csv"
) -> None:
    """
    Save timing metrics into a CSV file using pandas DataFrame.
    
    Each row is indexed by a timestamp and contains columns for various timing stages.
    
    Parameters:
    - averaging_and_median_time: Time for averaging and median filtering.
    - preprocessing_time: Time for preprocessing stage.
    - stitching_time: Time for stitching images.
    - marching_cubes_time: Time for Marching Cubes.
    - poisson_reconstruction_time: Time for Poisson surface reconstruction.
    - elapsed_time: Total elapsed time.
    - output_folder: Directory to save the CSV file.
    - csv_filename: Name of the CSV file.
    """
    os.makedirs(output_folder, exist_ok=True)
    full_path = os.path.join(output_folder, csv_filename)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create a DataFrame with the timing data
    new_row = pd.DataFrame({
        "averaging_and_median_time": [averaging_and_median_time],
        "preprocessing_time": [preprocessing_time],
        "stitching_time": [stitching_time],
        "marching_cubes_time": [marching_cubes_time],
        "poisson_reconstruction_time": [poisson_reconstruction_time],
        "elapsed_time": [elapsed_time]
    }, index=[timestamp])
    
    # Append to existing file or create new one
    if os.path.exists(full_path):
        df_existing = pd.read_csv(full_path, index_col=0)
        df_combined = pd.concat([df_existing, new_row])
    else:
        df_combined = new_row
    
    df_combined.to_csv(full_path)
    print(f"Timing data saved to: {full_path}")


def process_directory(base_dir):
    
    #STATS
    #time
    start_time = time.time()
    averaging_and_median_time= -1
    preprocessing_time=-1
    stitiching_time=-1
    marching_cubes_time=-1
    poisson_reconstruction_time=-1
    elapsed_time=-1
    #files
    number_of_valid_files = None
    
    # usar una carpeta particular 
    if base_dir:
      BASE_DIR = Path(base_dir)
      print(f"[PROCESS_DIRECTORY] Working with {base_dir}")
    
    AVERAGED_DIR = BASE_DIR / "averaged_images"
    AVERAGE_DIR = AVERAGED_DIR/"average"
    MEDIAN_DIR = AVERAGED_DIR/"median"
    PREPROCESSING_DIR = BASE_DIR/ "preprocessing" 
    GAUSS_DIR = PREPROCESSING_DIR/"output"
    STITCHED_DIR = BASE_DIR / "stitched"
    OUTPUTS_DIR = BASE_DIR / "outputs"
    STATS_DIR = BASE_DIR / "stats"
    
    # Asegurar que las carpetas necesarias existen
    Path(OUTPUTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(PREPROCESSING_DIR).mkdir(parents=True, exist_ok=True)
    Path(GAUSS_DIR).mkdir(parents=True, exist_ok=True)
    Path(AVERAGED_DIR).mkdir(parents=True, exist_ok=True)
    Path(STITCHED_DIR).mkdir(parents=True, exist_ok=True)
    Path(STATS_DIR).mkdir(parents=True, exist_ok=True)
    
    
    #AVERAGING AND MEDIANS
    
    if AVERAGES_AND_MEDIAN_DONE is False:
        print("🔹Grouping valid .tif files...")
        #count valid files
        ignore_lst = ["averaged_images"]
        number_of_valid_files = image_utilities.list_valid_files(BASE_DIR, recursively = True , ignore_lst = ignore_lst, ext= ".tif" )
        #create groups
        groups = None
        if not groups:
            groups = image_utilities.group_by_v_h(BASE_DIR, re_name = "volume_scan_re", ignore_lst =["averaged_images"], ext = ".tif" )
        if not groups:
            print("Assuming single volume scan")
            groups = image_utilities.group_by_v_h(BASE_DIR, re_name = "single_volume_scan", ignore_lst =["averaged_images"], ext = ".tif" )
        if not groups:
            print("❌ No se encontraron grupos de imágenes.")
            return
        print("🔹 Calculando promedios y medianas...")
        #avg_dir, median_dir = preprocessing.average_and_median_images(groups, BASE_DIR, AVERAGED_DIR)
        avg_dir, median_dir = preprocessing.average_and_median_images_cuda(groups, BASE_DIR, AVERAGED_DIR)
        if not avg_dir:
            print("❌ No se generaron imágenes promedio.")
            return
    else:
        median_dir = MEDIAN_DIR
        print(f"✅ Average images already processed, fetching images in {MEDIAN_DIR = }.")
    
    #time
    
    start_preprocessing_time = time.time()
    averaging_and_median_time = start_preprocessing_time - start_time
    
    print(f"Averaging and median time: {averaging_and_median_time}")
    collected = gc.collect()  # Free memory from averaging stage
    print(f"[GC] Unreachable objects collected: {collected}")
    
    #PREPROCESSING
    
    if PREPROCESSING_DONE is False:
        print("🔹 Realizando filtrado por renormalización y gaussiano por cada valor vertical...")
        gauss_dir, image_paths_by_h, image_paths_by_v = preprocessing.preprocessing_routine(folder_path=median_dir, preprocessing_dir = PREPROCESSING_DIR, re_name = "median_files_re")
        
        if not gauss_dir:
            print("❌ No se generaron imágenes filtradas.")
            return
    else:
        print("✅ Preprocessing done already, skipping.")
    

    start_stitiching_time = time.time()
    preprocessing_time = start_stitiching_time - start_preprocessing_time
        
    print(f"preprocessing_time: {preprocessing_time}")
    collected = gc.collect()  # Free memory from preprocessing
    print(f"[GC] Unreachable objects collected: {collected}")
        
    #STITCHING 
    if STITCHING_DONE is False:
        image_paths_by_h, image_paths_by_v = preprocessing.get_image_paths(GAUSS_DIR)
        
        optimal_overlap = 0
        try:
            optimal_overlap = get_optimal_overlap(GAUSS_DIR)
            if optimal_overlap:
                print("[STITCHING] Optimal overlap: ", optimal_overlap)
        except Exception as e:
            print(e)
            optimal_overlap = 0
        print("🔹 Realizando stitching horizontal por cada valor vertical...")
        
        #stitch_images(image_paths_by_v, GAUSS_DIR, STITCHED_DIR, method="simple_stitch",
        #              overlap_pixels=optimal_overlap, center_offset=0, skew=10)
        #stitch_images(image_paths_by_v, GAUSS_DIR, STITCHED_DIR, method="simple_stitch",
        #              overlap_pixels=optimal_overlap, center_offset=0, skew=10)      
        stitch_images(
                        image_paths_by_v=image_paths_by_v,
                        input_folder=GAUSS_DIR,
                        output_folder=STITCHED_DIR,
                        overlap_side="left",                # or "right", depending on your data
                        frameShift=0,                       # Optional: shift to tune overlap function
                        displacementShift=optimal_overlap, # Replaces overlap_pixels
                        overlap_function=get_overlaps_static,  # Or any custom model like get_overlaps
                        mark_boundaries=False              # Set True to visually debug overlaps
                    )

                      
    else:
        print("✅ Stitching done already, skipping.")
    
    start_marching_cubes_time = time.time()
    stitiching_time = start_marching_cubes_time - start_stitiching_time 
        
    print(f"stitiching_time: {stitiching_time}")
    collected = gc.collect()  # Free memory from stitching
    print(f"[GC] Unreachable objects collected: {collected}")
    
    #MARCHING CUBES
    temp_input_stl = os.path.join(OUTPUTS_DIR, "temp_input.stl")
        
    if MARCHING_CUBES_DONE is False:
        print("🔹 Construyendo volumen 3D a partir de imágenes stitchadas...")
        volume, slice_indices, max_h, max_w = load_and_build_volume(STITCHED_DIR)

        print("🔹 Ejecutando algoritmo Marching Cubes para generar malla 3D...")
        mesh = marching_cubes_to_trimesh(volume, slice_indices, max_h, max_w)

        
        mesh.export(temp_input_stl)
    else:
        print("✅ Marching cubes done already, skipping.")
    
    output_stl_path = os.path.join(OUTPUTS_DIR, "reconstructed_mesh.stl")
    collected = collected = gc.collect()  # Free memory from marching cubes
    print(f"[GC] Unreachable objects collected: {collected}")
    
    #POISSON_RECONSTRUCTION_DONE
    start_poisson_reconstruction_time = time.time()
    marching_cubes_time = start_poisson_reconstruction_time - start_marching_cubes_time 
        
    print(f"marching_cubes_time: {marching_cubes_time}")
    
    if POISSON_RECONSTRUCTION_DONE is False:
        print("🔹 Aplicando reconstrucción Poisson sobre la malla...")
        poisson_reconstruct_external(temp_input_stl, output_stl_path)
    
    end_time = time.time()
    poisson_reconstruction_time = end_time - start_poisson_reconstruction_time
        
    print(f"poisson_reconstruction_time: {poisson_reconstruction_time}")
    print("✅ Todo finalizado correctamente. Archivo guardado en:", output_stl_path)
    
    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    print(f"""{averaging_and_median_time=}\n{preprocessing_time=}\n{stitiching_time=}
        \n{marching_cubes_time=}\n{poisson_reconstruction_time=}\n{elapsed_time=}\n""")
        
    #save stats
    save_stats(
    averaging_and_median_time,
    preprocessing_time,
    stitching_time,
    marching_cubes_time,
    poisson_reconstruction_time,
    elapsed_time,
    valid_files, 
    output_folder = STATS_DIR
    )

def main():
    dirs_to_process = [ r"G:\morpho-2025-07-17",r"G:\foursamples"]
    for directory in dirs_to_process:
        process_directory(base_dir = directory)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        print(e)
        print(f"""{averaging_and_median_time=}\n{preprocessing_time=}\n{stitiching_time=}
        \n{marching_cubes_time=}\n{poisson_reconstruction_time=}\n{elapsed_time=}\n""")


import os
import pandas as pd
from datetime import datetime
