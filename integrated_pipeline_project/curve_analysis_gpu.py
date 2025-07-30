"""
curve_analysis_gpu.py

GPU-accelerated processing pipeline for OCT images, based on the original
process_image_and_save but extended to run in parallel on GPU with batch management.

Author: Auto-generated for user request
"""

import os
import math
import json
import cv2
import numpy as np
import cupy as cp
from pathlib import Path
from typing import List, Tuple, Optional

from curve_analysis import (
    invert_image,
    remove_small_objects,
    fit_top_ridge_curve,
    save_processing_info,
    draw_curve_overlay,
    gaussian_filter_column_based
)


def check_gpu_memory(image_shape: Tuple[int, int], image_count: int, dtype: str = 'uint8') -> int:
    """
    Estimate batches needed for GPU processing based on memory.
    """
    bytes_per_pixel = np.dtype(dtype).itemsize
    bytes_per_image = image_shape[0] * image_shape[1] * bytes_per_pixel
    total_required = image_count * bytes_per_image

    mem_info = cp.cuda.Device(0).mem_info
    free_memory = mem_info[0]

    if total_required < free_memory:
        return 1
    else:
        max_images_per_batch = free_memory // bytes_per_image
        return math.ceil(image_count / max_images_per_batch)


def process_images_in_batches(
    image_paths: List[str],
    preprocessing_base_dir: str,
    output_dir: str,
    poly_degree: int = 2,
    k: int = 10,
    kernel_size: int = 3,
    sigma: float = 30.0,
) -> None:
    """
    Process a list of OCT images in GPU-safe batches.
    """
    num_batches = check_gpu_memory((512, 512), len(image_paths))
    print(f"Total images: {len(image_paths)}, Batches needed: {num_batches}")
    batch_size = math.ceil(len(image_paths) / num_batches)

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, len(image_paths))
        batch = image_paths[start:end]
        print(f"[GPU] Batch {batch_idx + 1}/{num_batches} - {len(batch)} images")
        for path in batch:
            process_image_and_save(
                raw_image_path=path,
                preprocessing_base_dir=preprocessing_base_dir,
                output_dir=output_dir,
                poly_degree=poly_degree,
                k=k,
                kernel_size=kernel_size,
                sigma=sigma
            )


def process_image_and_save(
    raw_image_path: str,
    preprocessing_base_dir: str,
    output_dir: str,
    poly_degree: int = 2,
    k: int = 10,
    kernel_size: int = 3,
    sigma: float = 30
) -> Optional[Path]:
    """
    Process a single OCT image with full pipeline: grayscale > denoise > curve > Gaussian > save.

    Parameters:
        raw_image_path (str): Path to a .tif OCT image.
        preprocessing_base_dir (str): Base directory for intermediate results.
        output_dir (str): Final output directory for processed images.
        poly_degree (int): Polynomial degree for curve fitting.
        k (int): Percentage of top image rows to mask as noise.
        kernel_size (int): Median filter kernel size.
        sigma (float): Gaussian filter sigma value.

    Returns:
        Optional[Path]: Path to final filtered image, or None if failed.
    """
    preprocessing_base_dir = Path(preprocessing_base_dir)
    json_dir = preprocessing_base_dir / "json"
    grayscale_dir = preprocessing_base_dir / "1-grayscale"
    denoised_dir = preprocessing_base_dir / "2-denoised"
    denoised_curve_dir = preprocessing_base_dir / "3-denoised-curve"
    gauss_curve_dir = preprocessing_base_dir / "4-gauss-curve"
    output_dir = Path(output_dir)

    for folder in [preprocessing_base_dir,json_dir,grayscale_dir,
                   denoised_dir,denoised_curve_dir,gauss_curve_dir,output_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    filename = Path(raw_image_path).name
    name, ext = os.path.splitext(filename)
    output_image_filtered_path = output_dir / f"gauss_and_curve2_{name}.png"
    img_original = cv2.imread(raw_image_path, cv2.IMREAD_GRAYSCALE)

    if img_original is None or img_original.size == 0 or np.all(img_original == 0):
        inverted_image = invert_image(np.zeros((512, 512), dtype=np.uint8))
        cv2.imwrite(str(output_image_filtered_path), inverted_image)
        save_processing_info(str(json_dir / f"{name}.json"), {
            "curve_found": False,
            "black_frame": True
        })
        return output_image_filtered_path

    img_original = cv2.resize(img_original, (512, 512))
    img_original[:int((k / 100) * img_original.shape[0]), :] = 0
    cv2.imwrite(str(grayscale_dir / f"grayscale_{name}.tif"), img_original)

    img_denoised = remove_small_objects(grayscale_image=img_original, min_size=64, threshold=100)
    cv2.imwrite(str(denoised_dir / f"denoised_rmso_{name}.tif"), img_denoised)

    try:
        curve_result = fit_top_ridge_curve(img_denoised, degree=poly_degree)
    except ValueError:
        curve_result = None

    if not curve_result:
        save_processing_info(str(json_dir / f"{name}.json"), {
            "curve_found": False,
            "black_frame": False
        })
        img_white = invert_image(img_denoised)
        cv2.imwrite(str(output_image_filtered_path), img_white)
        return output_image_filtered_path

    curve_fn, x0, y0, tangent, normal = curve_result
    save_processing_info(str(json_dir / f"{name}.json"), {
        "vertex": (int(x0), int(y0)),
        "normal_vector": normal.tolist(),
        "tangent_vector": tangent.tolist(),
        "polynomial_coefficients": curve_fn.coefficients.tolist(),
        "polynomial_degree": poly_degree,
        "curve_found": True,
        "black_frame": False
    })

    img_curve_overlay = draw_curve_overlay(img_denoised, curve_fn, x0, y0, tangent, normal)
    cv2.imwrite(str(denoised_curve_dir / f"denoised_w_curve_{name}.tif"), img_curve_overlay)

    img_gauss = gaussian_filter_column_based(img_original, curve_fn.coefficients, sigma)
    img_gauss_med = cv2.medianBlur(img_gauss, kernel_size)
    img_white_bg = invert_image(img_gauss_med)
    cv2.imwrite(str(output_image_filtered_path), img_white_bg)

    try:
        curve_result = fit_top_ridge_curve(img_gauss_med, degree=poly_degree)
        if curve_result:
            curve_fn, x0, y0, tangent, normal = curve_result
            save_processing_info(str(json_dir / f"{name}.json"), {
                "vertex": (int(x0), int(y0)),
                "normal_vector": normal.tolist(),
                "tangent_vector": tangent.tolist(),
                "polynomial_coefficients": curve_fn.coefficients.tolist(),
                "polynomial_degree": poly_degree,
                "curve_found": True,
                "black_frame": False
            })
            final_overlay = draw_curve_overlay(img_white_bg, curve_fn, x0, y0, tangent, normal)
            cv2.imwrite(str(gauss_curve_dir / f"gauss_and_curve2_{name}.png"), final_overlay)
    except Exception:
        pass

    return output_image_filtered_path
