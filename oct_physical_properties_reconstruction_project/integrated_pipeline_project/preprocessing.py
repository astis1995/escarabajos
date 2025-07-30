import os
#from curve_analysis import process_image_and_save
import curve_analysis_gpu #import process_image_and_save  # usa el canvas actualizado
from image_utilities import group_by_v_h
from collections import defaultdict
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

def compute_average_image(images, preserve_bitdepth=False):
    """Per-pixel average of a list of images (NumPy arrays), with better handling of bit-depth."""
    
    # Ensure images are in float for proper averaging
    avg_img = np.mean(images, axis=0)
    
    if preserve_bitdepth:
        # If you want to preserve higher bit-depths (e.g., uint16), return the average as uint16
        avg_img = np.clip(avg_img, 0, 65535).astype(np.uint16)  # or `np.uint16` to preserve range
    else:
        # Clip and convert to uint8 for images in that bit-depth
        avg_img = np.clip(avg_img, 0, 255).astype(np.uint8)

    return avg_img

def compute_median_image(images):
    """Median image based on total number of pixels (not pixel-wise median)."""
    # Sort images by number of pixels (width * height)
    sorted_images = sorted(images, key=lambda img: img.shape[0] * img.shape[1])
    n = len(sorted_images)
    if n % 2 == 1:
        return sorted_images[n // 2].astype(np.uint8)
    else:
        # Average the two median images
        img1 = sorted_images[n // 2 - 1]
        img2 = sorted_images[n // 2]
        avg_img = ((img1.astype(np.float32) + img2.astype(np.float32)) / 2.0)
        return np.clip(avg_img, 0, 255).astype(np.uint8)
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
import time

def get_grouped_dict(directory , re_name = "volume_scan_re", ignore_lst = None, ext = None):
    
    testing = False
    
    
    directory = Path(directory) 
    output_dict = defaultdict(list)
    
    grouped_dict = group_by_v_h(directory, re_name, ignore_lst, ext)
    #if testing: print("grouped_dict",grouped_dict)
    
    for (v, h), file_list in grouped_dict.items():
        images = []
        paths = []
        for path in file_list:
            #print("Current path", path)
            filename = Path(path).name
            try:
                img = Image.open(path).convert('L')
                #images.append(np.array(img, dtype=np.float32))
                images.append(np.array(img, dtype=np.float16))
                paths.append(path)
            except Exception as e:
                print(f"Failed to load {filename}: {e}")
        if not images:
            print("No images loaded, check", (v, h), file_list )
        
        output_dict[v,h]
        
        #now for each (v, h) 
        file_sub_dict = {
                        "images":images,
                        "paths":paths,
                            }
        output_dict[v,h] = file_sub_dict
        

    return output_dict

from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Union
import cupy as cp
import numpy as np
from PIL import Image


def compute_average_image_cuda(images: List[cp.ndarray]) -> cp.ndarray:
    """
    Compute the per-pixel average of a list of grayscale images using GPU.

    Parameters:
    - images (List[cp.ndarray]): List of 2D CuPy arrays of shape (H, W)

    Returns:
    - cp.ndarray: 2D CuPy array (H, W) with dtype=uint8 representing average image
    """
    stack = cp.stack(images, axis=0)
    avg = cp.mean(stack, axis=0)
    return cp.clip(avg, 0, 255).astype(cp.uint8)


def compute_median_image_cuda(images: List[cp.ndarray]) -> cp.ndarray:
    """
    Compute the per-pixel median of a list of grayscale images using GPU.

    Parameters:
    - images (List[cp.ndarray]): List of 2D CuPy arrays of shape (H, W)

    Returns:
    - cp.ndarray: 2D CuPy array (H, W) with dtype=uint8 representing median image
    """
    stack = cp.stack(images, axis=0)
    med = cp.median(stack, axis=0)
    return cp.clip(med, 0, 255).astype(cp.uint8)


def average_and_median_images_cuda(
    grouped_dict: Dict[Tuple[int, int], List[str]],
    base_folder: Union[str, Path],
    output_folder: Union[str, Path]
) -> Tuple[Path, Path]:
    """
    Compute average and median grayscale images by group, using CUDA acceleration.

    Parameters:
    - grouped_dict (Dict[Tuple[int, int], List[str]]): Dictionary mapping (v, h) keys to lists of image filenames.
    - base_folder (str or Path): Directory containing input images.
    - output_folder (str or Path): Directory where output will be saved.

    Returns:
    - Tuple[Path, Path]: Paths to the folders containing average and median images respectively.
    """

    base_folder = Path(base_folder)
    output_folder = Path(output_folder)

    avg_dir = output_folder / "average"
    med_dir = output_folder / "median"
    avg_dir.mkdir(parents=True, exist_ok=True)
    med_dir.mkdir(parents=True, exist_ok=True)

    averaged_paths = defaultdict(list)
    median_paths = defaultdict(list)

    for (v, h), file_list in grouped_dict.items():
        images = []
        for fname in file_list:
            path = base_folder / fname
            try:
                img = Image.open(path).convert("L")
                np_img = np.array(img, dtype=np.float32)
                cp_img = cp.asarray(np_img)
                images.append(cp_img)
            except Exception as e:
                print(f"Failed to load {fname}: {e}")
        
        if not images:
            continue

        avg_img = compute_average_image_cuda(images)
        med_img = compute_median_image_cuda(images)

        avg_pil = Image.fromarray(cp.asnumpy(avg_img))
        med_pil = Image.fromarray(cp.asnumpy(med_img))

        avg_filename = f"{v:02d}-{h:02d}_avg.tif"
        med_filename = f"{v:02d}-{h:02d}_med.tif"
        avg_path = avg_dir / avg_filename
        med_path = med_dir / med_filename

        avg_pil.save(avg_path)
        med_pil.save(med_path)

        averaged_paths[v].append((h, str(avg_path)))
        median_paths[v].append((h, str(med_path)))

    return avg_dir, med_dir


def average_and_median_images(grouped_dict, base_folder, output_folder):
    base_folder = Path(base_folder)
    output_folder = Path(output_folder)

    avg_dir = output_folder / "average"
    med_dir = output_folder / "median"

    avg_dir.mkdir(parents=True, exist_ok=True)
    med_dir.mkdir(parents=True, exist_ok=True)

    averaged_paths = defaultdict(list)
    median_paths = defaultdict(list)

    for (v, h), file_list in grouped_dict.items():
        images = []
        for fname in file_list:
            path = base_folder / fname
            try:
                img = Image.open(path).convert('L')
                images.append(np.array(img, dtype=np.float32))
            except Exception as e:
                print(f"Failed to load {fname}: {e}")
        if not images:
            continue

        avg_img = compute_average_image(images)
        med_img = compute_median_image(images)

        avg_pil = Image.fromarray(avg_img)
        med_pil = Image.fromarray(med_img)

        avg_filename = f"{v:02d}-{h:02d}_avg.tif"
        med_filename = f"{v:02d}-{h:02d}_med.tif"

        avg_path = avg_dir / avg_filename
        med_path = med_dir / med_filename

        avg_pil.save(avg_path)
        med_pil.save(med_path)

        averaged_paths[v].append((h, str(avg_path)))
        median_paths[v].append((h, str(med_path)))

    return avg_dir, med_dir

def get_image_paths(processed_img_path):
    testing = False
    
    base_folder = processed_img_path
    if testing: print("processed_img_path", processed_img_path)
    #input_paths 
    image_paths_by_h = defaultdict(list)
    image_paths_by_v = defaultdict(list)
    
    grouped = group_by_v_h(base_folder, re_name = "gauss_files_re",
                                ignore_lst = None, recursive = False, ext = ".png")
                                
    if testing:  print("grouped", grouped)
    for (v, h), path_list in grouped.items():
        if testing:  print(v, h, path_list)
        image_paths_by_h[h].append((v, path_list[0]))
        image_paths_by_v[v].append((h, path_list[0]))
        
    return image_paths_by_h, image_paths_by_v
    
def preprocessing_routine(folder_path, preprocessing_dir, re_name="median_files_re"):
    """
    Procesa todas las imágenes de una carpeta aplicando preprocesamiento,
    reducción de ruido, análisis de curva y filtrado gaussiano.

    Parámetros:
    - folder_path: ruta de la carpeta con las imágenes
    """
    testing = False
    if not os.path.exists(folder_path):
        raise ValueError(f"[PREPROCESSING] Carpeta no encontrada: {folder_path}")
    
    # Get grouped dict from folder
    grouped_dict = get_grouped_dict(folder_path, re_name, ignore_lst=["preprocessing"], ext=".tif")
    
    # Create output dirs
    output_dir = preprocessing_dir / "output"
    os.makedirs(preprocessing_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Store processed image paths
    image_paths_by_h = defaultdict(list)
    image_paths_by_v = defaultdict(list)

    print(f"[PREPROCESSING] 📁 Procesando carpeta: {folder_path}")

    for (v, h), file_sub_dict in grouped_dict.items():
        path_obj = Path(file_sub_dict["paths"][0])
        fname = path_obj.name
        path = str(path_obj)

        if path.endswith(('.png', '.jpg', '.jpeg', '.tif')):
            img_path = path
            
            try:
                # --- Check if already processed ---
                expected_output_filename = get_processed_image_filename(img_path)
                expected_output_path = output_dir / expected_output_filename

                if expected_output_path.exists():
                    if testing:
                        print(f"[PREPROCESSING] ⏩ Ya procesado, saltando: {expected_output_filename.name}")

                    image_paths_by_h[h].append((v, str(expected_output_path)))
                    image_paths_by_v[v].append((h, str(expected_output_path)))
                    continue

                # --- Process image ---
                processed_img_path = curve_analysis_gpu.process_image_and_save(
                    img_path,
                    preprocessing_dir,
                    output_dir,
                    poly_degree=2,
                    k=10,
                    kernel_size=3,
                    sigma=60
                )

                if testing:
                    print(f"[PREPROCESSING] ✅ Procesado: {fname}")

                image_paths_by_h[h].append((v, processed_img_path))
                image_paths_by_v[v].append((h, processed_img_path))

            except Exception as e:
                import traceback
                print(f"[PREPROCESSING] ❌ Error al procesar {fname}: {e}")
                traceback.print_exc()
                with open("error_log.txt", "a") as log_file:
                    log_file.write(f"[PREPROCESSING] Error al procesar {fname}: {e}\n")
                    traceback.print_exc(file=log_file)

    return output_dir, image_paths_by_h, image_paths_by_v
    

def load_and_preprocess_image(filepath, k=10):
    """
    Carga una imagen en escala de grises, convierte a negro los pixeles del top k%,
    luego redimensiona a 512x512.

    Parámetros:
    - filepath: ruta del archivo de imagen
    - k: porcentaje superior de la imagen a convertir en negro (0-100)

    Retorna:
    - img_gray: imagen preprocesada (np.ndarray)
    """
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"[PREPROCESSING] No se pudo cargar la imagen: {filepath}")

    height = img.shape[0]
    top_k_pixels = int((k / 100) * height)

    # Convertir a negro la parte superior k%
    img[:top_k_pixels, :] = 0

    # Ahora redimensionar
    img = cv2.resize(img, (512, 512))

    return img
    


def noise_reduction(img, kernel_size=3):
    """
    Aplica un filtro de mediana para reducir ruido en la imagen.

    ¿Cómo funciona?
    Para cada píxel, considera una grilla (ventana) de tamaño kernel_size x kernel_size,
    ordena los píxeles de esa grilla de menor a mayor, toma el valor mediano,
    y sustituye el píxel central por ese valor.

    Parámetros:
    - img: imagen en escala de grises (np.ndarray)
    - kernel_size: tamaño de la grilla (debe ser impar, por ejemplo, 3, 5, 7)

    Retorna:
    - img_denoised: imagen sin ruido (np.ndarray)
    """

    img_denoised = cv2.medianBlur(img, kernel_size)
    return img_denoised
