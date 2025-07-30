import cv2
import numpy as np
import json
import os
import re
from PIL import Image
from scipy.ndimage import uniform_filter
from pathlib import Path
import warnings

def fit_top_ridge_curve(img_gray, degree=2):
    #print("fit top ridge, shape", img_gray.shape)
    h, w = img_gray.shape
    points_x, points_y = [], []

    #get the topmost pixel 
    for x in range(w):
        column = img_gray[:, x]
        y_indices = np.where(column > 5)[0]
        if len(y_indices) > 0:
            y_top = y_indices[0]
            points_x.append(x)
            points_y.append(y_top)

    points_x = np.array(points_x)
    points_y = np.array(points_y)
    
    if len(points_x) == 0 or len(points_y) == 0:
        raise ValueError("No points found in the image for fitting the curve.")
        
    #print("points x", len(points_x), "points_y", len(points_y))

    try:
        with warnings.catch_warnings(record=True) as warning:
            warnings.simplefilter('always')
            coeffs = np.polyfit(points_x, points_y, degree)
            if len(warning) > 0:
                print(f"[WARNING] Polyfit warning: {warning[-1].message}")
                return None
    except Exception as e:
        import traceback
        print(f"[PREPROCESSING] Error al procesar {img_gray.shape if hasattr(img_gray, 'shape') else 'unknown'}: {e}")
        traceback.print_exc()
        with open("error_log.txt", "a") as log_file:
            log_file.write(f"[PREPROCESSING] Error al procesar polyfit: {e}")

            traceback.print_exc(file=log_file)
        return None

    curve_fn = np.poly1d(coeffs)

    x_grid = np.linspace(0, w - 1, 1000)
    y_grid = curve_fn(x_grid)
    min_index = np.argmin(y_grid)
    x0 = x_grid[min_index]
    y0 = y_grid[min_index]

    dx = 1e-3
    dy = (curve_fn(x0 + dx) - curve_fn(x0 - dx)) / (2 * dx)
    tangent = np.array([1, dy]) / np.linalg.norm([1, dy])
    normal = np.array([-dy, 1]) / np.linalg.norm([-dy, 1])

    return curve_fn, x0, y0, tangent, normal

 
def gaussian_filter_column_based(img, poly_coeffs, sigma):
    filtered_img = np.zeros_like(img)
    rows, cols = img.shape

    for x in range(cols):
        y_center = int(np.polyval(poly_coeffs, x))
        for y in range(rows):
            gaussian_weight = np.exp(-((y - y_center) ** 2) / (2 * sigma ** 2))
            filtered_img[y, x] = img[y, x] * gaussian_weight

    return filtered_img.astype(np.uint8)

def save_processing_info(output_path, info):
    with open(output_path, 'w') as file:
        json.dump(info, file, indent=4)

def draw_curve_overlay(img, curve_fn, x0, y0, tangent, normal):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.circle(img_rgb, (int(x0), int(y0)), 4, (0, 0, 255), -1)
    cv2.line(img_rgb, (int(x0), int(y0)), (int(x0 + 40 * tangent[0]), int(y0 + 40 * tangent[1])), (255, 0, 0), 2)
    cv2.line(img_rgb, (int(x0), int(y0)), (int(x0 + 40 * normal[0]), int(y0 + 40 * normal[1])), (0, 255, 0), 2)

    for x in range(511):
        y1, y2 = int(curve_fn(x)), int(curve_fn(x + 1))
        if 0 <= y1 < 512 and 0 <= y2 < 512:
            cv2.line(img_rgb, (x, y1), (x + 1, y2), (255, 255, 0), 1)

    return img_rgb

def local_mean_image(grayscale_image, window_size=3):
    """
    Calcula la imagen del promedio local de vecinos para cada píxel.

    Input:
    - grayscale_image (np.ndarray): imagen en escala de grises
    - window_size (int): tamaño de la ventana (debe ser impar)

    Output:
    - mean_image (np.ndarray): imagen con el valor promedio local en cada píxel
    """
    return uniform_filter(grayscale_image.astype(np.float32), size=window_size)
    
def remove_small_objects(grayscale_image, min_size=64, threshold=100, mean_window=3):
    """
    Elimina objetos brillantes pequeños de una imagen en escala de grises, conservando el rango dinámico de los objetos grandes,
    con binarización basada en el promedio local de vecinos.

    Input:
    - grayscale_image (np.ndarray): imagen en escala de grises (fondo oscuro, detalles brillantes)
    - min_size (int): área mínima en píxeles para conservar un objeto brillante
    - threshold (int): umbral de promedio local para binarizar
    - mean_window (int): tamaño de ventana para el promedio de vecinos

    Output:
    - filtered_image (np.ndarray): imagen con objetos pequeños eliminados (convertidos en negro), conservando el resto del contenido original
    """
    # Calcular imagen de promedios locales
    local_avg = local_mean_image(grayscale_image, window_size=mean_window)

    # Binarizar: píxeles cuyo promedio local >= threshold
    binary = np.where(local_avg >= threshold, 255, 0).astype(np.uint8)

    # Etiquetar componentes conectadas
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    # Crear máscara para conservar solo objetos grandes
    mask = np.zeros_like(grayscale_image, dtype=np.uint8)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_size:
            mask[labels == i] = 1  # usar 1 para conservar multiplicación posterior

    # Aplicar la máscara sobre la imagen original para conservar el rango dinámico
    filtered = (grayscale_image * mask).astype(np.uint8)
    return filtered

    
import warnings
from typing import Optional

def invert_image(original_image):
    if original_image.dtype != np.uint8:
            raise ValueError("Image must be of dtype uint8 (8-bit grayscale).")
        
    inverted_image = 255 - original_image
    return inverted_image

from pathlib import Path

def get_processed_image_filename(raw_image_path, prefix="gauss_and_curve2_", ext=".png"):
    """
    Generates the consistent output filename used for processed images.
    
    Parameters:
    - raw_image_path: Path or string to the raw image
    - prefix: optional prefix for output filename
    - ext: desired file extension for output image
    
    Returns:
    - Path object: output filename (not full path, just the filename)
    """
    name = Path(raw_image_path).stem
    return Path(f"{prefix}{name}{ext}")

def process_image_and_save(raw_image_path, preprocessing_base_dir, output_dir, poly_degree=2, k=10, kernel_size=3, sigma=30):
    """
    raw_image_path: path to a .tif OCT image blackish background, whiteish details
    
    returns:
    output_dir: a directory where all processed .tif are located and each has a 
    white bg and black details, ready to be used in marching cubes
    """
    
    preprocessing_base_dir = Path(preprocessing_base_dir)
    json_dir = preprocessing_base_dir / "json"
    grayscale_dir = preprocessing_base_dir / "1-grayscale"
    denoised_dir = preprocessing_base_dir / "2-denoised"
    denoised_curve_dir = preprocessing_base_dir / "3-denoised-curve"
    gauss_curve_dir = preprocessing_base_dir / "4-gauss-curve"
    output_dir = Path(output_dir)
    
    folders_to_create = [preprocessing_base_dir,json_dir,grayscale_dir,
    denoised_dir,denoised_curve_dir, gauss_curve_dir,output_dir]
    
    for folder in folders_to_create:
        if not os.path.exists(folder):
            os.makedirs(folder)
        
    #step 1.1: load original image. black bg
    filename = Path(raw_image_path).name
    name = Path(raw_image_path).stem
    output_image_filtered_path = output_dir / get_processed_image_filename(raw_image_path)
    img_original = cv2.imread(raw_image_path, cv2.IMREAD_GRAYSCALE)
    
    #step 1.2: check black frame and save to 
    is_all_black = np.all(img_original == 0)
    print("Processing:", filename, name, "is_all_black", is_all_black)
    #print("Original dimensions:", img_original.shape)

    black_frame = (
        img_original is None or
        img_original.size == 0 or
        np.all(img_original == 0)
    )
    #print("black_frame", black_frame, f"{img_original is None=}|{img_original.size == 0=}|{np.all(img_original == 0)=}" )

    if black_frame:
        #step 1.3a: invert image white bg and save
        inverted_image = invert_image(img_original)
        #save a white frame
        cv2.imwrite(output_image_filtered_path, inverted_image)
        info = {"curve_found": False, "black_frame": True}
        save_processing_info(os.path.join(preprocessing_base_dir, name + '.json'), info)
        return output_image_filtered_path
    
    #step 1.3b not black frame? delete top noise from black bg original
    img_original = cv2.resize(img_original, (512, 512))
    top_k = int((k / 100) * img_original.shape[0])
    img_original[:top_k, :] = 0

    #save grayscale image without top noise
    cv2.imwrite(grayscale_dir / f"grayscale_{name}.tif", img_original)
    
    #step 2.0: denoise remove small objects and save, black bg
    img_denoised = remove_small_objects(grayscale_image=img_original, min_size=64*1, threshold=100)
    cv2.imwrite(denoised_dir/ f"denoised_rmso_{name}.tif", img_denoised)

    #step 2.1: try to fit curve, black bg needed
    no_white_points_flag = False
    
    try:
        curve_result = fit_top_ridge_curve(img_denoised, degree=poly_degree)
    except ValueError as e: 
        print(e)
        no_white_points_flag = True
        
    #step 2.2a: no curve could be found
    if no_white_points_flag:
        print("no_white_points_flag")
        info = {"curve_found": False, "black_frame": False}
        save_processing_info(os.path.join(json_dir, name + '.json'), info)

        # Guardar la imagen ya filtrada para mantener consistencia 
        # cambiar fondo a blanco 
        img_denoised_white_bg = invert_image(img_denoised)
        cv2.imwrite(output_image_filtered_path, img_denoised_white_bg)
        return output_image_filtered_path
    
    #step 2.2b:  a curve could be found
    if curve_result:
        #step 2.2b1: add curve info to json
        curve_fn, x0, y0, tangent, normal = curve_result
        info = {
            "vertex": (int(x0), int(y0)),
            "normal_vector": normal.tolist(),
            "tangent_vector": tangent.tolist(),
            "polynomial_coefficients": curve_fn.coefficients.tolist(),
            "polynomial_degree": poly_degree,
            "curve_found": True,
            "black_frame": False
        }
        save_processing_info(os.path.join(json_dir, name + '.json'), info)
    
    #step 2.2b2: save denoised image and curve in the same image 
    #draw
    denoised_with_curve = draw_curve_overlay(img_denoised, curve_fn, x0, y0, tangent, normal)
    #save 
    cv2.imwrite(denoised_curve_dir/ f"denoised_w_curve_{name}.tif", denoised_with_curve)
    
    #step 3.0 gauss filtering
    img_gauss_filtered = gaussian_filter_column_based(img_original, curve_fn.coefficients, sigma)
    img_gauss_filtered_renormalized = cv2.medianBlur(img_gauss_filtered, kernel_size)
    
    #step 3.1 invert to white bg and save
    img_gauss_filtered_renormalized_white_bg = invert_image(img_gauss_filtered_renormalized)
    cv2.imwrite(output_image_filtered_path, img_gauss_filtered_renormalized_white_bg)

    #step 3.2 try to get curve again, black bg
    curve_result = fit_top_ridge_curve(img_gauss_filtered_renormalized, degree=poly_degree)
    
    #step 3.3a A curve was found
    if curve_result:
        curve_fn, x0, y0, tangent, normal = curve_result
        info = {
            "vertex": (int(x0), int(y0)),
            "normal_vector": normal.tolist(),
            "tangent_vector": tangent.tolist(),
            "polynomial_coefficients": curve_fn.coefficients.tolist(),
            "polynomial_degree": poly_degree,
            "curve_found": True,
            "black_frame": False
        }
        save_processing_info(os.path.join(json_dir, name + '.json'), info)
        
        #directory 
        img_gauss_w_curve_path = gauss_curve_dir / get_processed_image_filename(raw_image_path)

        #superpose curve
        gauss_with_curve = draw_curve_overlay(img_gauss_filtered_renormalized_white_bg, curve_fn, x0, y0, tangent, normal)
        cv2.imwrite(img_gauss_w_curve_path, gauss_with_curve)

    return output_image_filtered_path


  
def process_image_and_save_old(raw_image_path, preprocessing_base_dir, output_dir, poly_degree=2, k=10, kernel_size=3, sigma=30):
    """
    raw_image_path: path to a .tif OCT image blackish background, whiteish details
    
    returns:
    output_dir: a directory where all processed .tif are located and each has a 
    white bg and black details, ready to be used in marching cubes
    """
    
    preprocessing_base_dir = Path(preprocessing_base_dir)
    json_dir = preprocessing_base_dir / "json"
    grayscale_dir = preprocessing_base_dir / "1-grayscale"
    denoised_dir = preprocessing_base_dir / "2-denoised"
    denoised_curve_dir = preprocessing_base_dir / "3-denoised-curve"
    gauss_curve_dir = preprocessing_base_dir / "4-gauss-curve"
    output_dir = Path(output_dir)
    
    folders_to_create = [preprocessing_base_dir,json_dir,grayscale_dir,
    denoised_dir,denoised_curve_dir, gauss_curve_dir,output_dir]
    
    for folder in folders_to_create:
        if not os.path.exists(folder):
            os.makedirs(folder)
        
    #step 1.1: load original image. black bg
    filename = Path(raw_image_path).name
    name, ext = os.path.splitext(filename)
    output_image_filtered_path = output_dir/ f"gauss_and_curve2_{name}.png"
    img_original = cv2.imread(raw_image_path, cv2.IMREAD_GRAYSCALE)
    
    #step 1.2: check black frame and save to 
    is_all_black = np.all(img_original == 0)
    print("Processing:", filename, name, "is_all_black", is_all_black)
    #print("Original dimensions:", img_original.shape)

    black_frame = (
        img_original is None or
        img_original.size == 0 or
        np.all(img_original == 0)
    )
    #print("black_frame", black_frame, f"{img_original is None=}|{img_original.size == 0=}|{np.all(img_original == 0)=}" )

    if black_frame:
        #step 1.3a: invert image white bg and save
        inverted_image = invert_image(img_original)
        #save a white frame
        cv2.imwrite(output_image_filtered_path, inverted_image)
        info = {"curve_found": False, "black_frame": True}
        save_processing_info(os.path.join(preprocessing_base_dir, name + '.json'), info)
        return output_image_filtered_path
    
    #step 1.3b not black frame? delete top noise from black bg original
    img_original = cv2.resize(img_original, (512, 512))
    top_k = int((k / 100) * img_original.shape[0])
    img_original[:top_k, :] = 0

    #save grayscale image without top noise
    cv2.imwrite(grayscale_dir / f"grayscale_{name}.tif", img_original)
    
    #step 2.0: denoise remove small objects and save, black bg
    img_denoised = remove_small_objects(grayscale_image=img_original, min_size=64*1, threshold=100)
    cv2.imwrite(denoised_dir/ f"denoised_rmso_{name}.tif", img_denoised)

    #step 2.1: try to fit curve, black bg needed
    no_white_points_flag = False
    
    try:
        curve_result = fit_top_ridge_curve(img_denoised, degree=poly_degree)
    except ValueError as e: 
        print(e)
        no_white_points_flag = True
        
    #step 2.2a: no curve could be found
    if no_white_points_flag:
        print("no_white_points_flag")
        info = {"curve_found": False, "black_frame": False}
        save_processing_info(os.path.join(json_dir, name + '.json'), info)

        # Guardar la imagen ya filtrada para mantener consistencia 
        # cambiar fondo a blanco 
        img_denoised_white_bg = invert_image(img_denoised)
        cv2.imwrite(output_image_filtered_path, img_denoised_white_bg)
        return output_image_filtered_path
    
    #step 2.2b:  a curve could be found
    if curve_result:
        #step 2.2b1: add curve info to json
        curve_fn, x0, y0, tangent, normal = curve_result
        info = {
            "vertex": (int(x0), int(y0)),
            "normal_vector": normal.tolist(),
            "tangent_vector": tangent.tolist(),
            "polynomial_coefficients": curve_fn.coefficients.tolist(),
            "polynomial_degree": poly_degree,
            "curve_found": True,
            "black_frame": False
        }
        save_processing_info(os.path.join(json_dir, name + '.json'), info)
    
    #step 2.2b2: save denoised image and curve in the same image 
    #draw
    denoised_with_curve = draw_curve_overlay(img_denoised, curve_fn, x0, y0, tangent, normal)
    #save 
    cv2.imwrite(denoised_curve_dir/ f"denoised_w_curve_{name}.tif", denoised_with_curve)
    
    #step 3.0 gauss filtering
    img_gauss_filtered = gaussian_filter_column_based(img_original, curve_fn.coefficients, sigma)
    img_gauss_filtered_renormalized = cv2.medianBlur(img_gauss_filtered, kernel_size)
    
    #step 3.1 invert to white bg and save
    img_gauss_filtered_renormalized_white_bg = invert_image(img_gauss_filtered_renormalized)
    cv2.imwrite(output_image_filtered_path, img_gauss_filtered_renormalized_white_bg)

    #step 3.2 try to get curve again, black bg
    curve_result = fit_top_ridge_curve(img_gauss_filtered_renormalized, degree=poly_degree)
    
    #step 3.3a A curve was found
    if curve_result:
        curve_fn, x0, y0, tangent, normal = curve_result
        info = {
            "vertex": (int(x0), int(y0)),
            "normal_vector": normal.tolist(),
            "tangent_vector": tangent.tolist(),
            "polynomial_coefficients": curve_fn.coefficients.tolist(),
            "polynomial_degree": poly_degree,
            "curve_found": True,
            "black_frame": False
        }
        save_processing_info(os.path.join(json_dir, name + '.json'), info)
        
        #directory 
        img_gauss_w_curve_path = gauss_curve_dir / f"gauss_and_curve2_{name}.png"
        #superpose curve
        gauss_with_curve = draw_curve_overlay(img_gauss_filtered_renormalized_white_bg, curve_fn, x0, y0, tangent, normal)
        cv2.imwrite(img_gauss_w_curve_path, gauss_with_curve)

    return output_image_filtered_path
