

import cupy as cp
import cv2
import os

test_dir = r"G:\pluma-2025-07-14 (2)\test"

import os
import cv2
import re
from typing import List, Tuple
import numpy as np

def load_images_with_indices(test_dir: str) -> List[Tuple[int, int, np.ndarray]]:
    """
    Loads grayscale images from a directory and parses vertical (v) and horizontal (h)
    indices from filenames of the format: v-h_suffix.tif (e.g., 1535-13_med.tif).

    Parameters
    ----------
    test_dir : str
        Path to the directory containing the images.

    Returns
    -------
    v_h_image_tuple : list of tuple[int, int, numpy.ndarray]
        A list of (v_index, h_index, image) tuples. Each image is loaded in grayscale
        using cv2.IMREAD_GRAYSCALE.

    Notes
    -----
    - Only files matching the pattern \d+-\d+_*.tif are processed.
    - Filenames not matching the expected format are skipped with a warning.
    - Unreadable images will be skipped with a warning.
    """
    v_h_image_tuple = []

    pattern = re.compile(r".*(\d+)-(\d+)_.*\.(tif)*(png)*$", re.IGNORECASE)

    for filename in os.listdir(test_dir):
        match = pattern.match(filename)
        if match:
            v_index = int(match.group(1))
            h_index = int(match.group(2))

            path = os.path.join(test_dir, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            if img is not None:
                v_h_image_tuple.append((v_index, h_index, img))
            else:
                print(f"[Warning] Could not load image: {filename}")
        else:
            print(f"[Skipped] Filename does not match pattern: {filename}")

    return v_h_image_tuple

import cupy as cp
from typing import List, Tuple
import numpy as np

def extract_first_h0_images_to_gpu(v_h_image_tuple: List[Tuple[int, int, np.ndarray]]) -> dict[int, cp.ndarray]:
    """
    Extracts the first image where h == 0 for each unique v index,
    and converts it to a CuPy array for GPU processing.

    Parameters
    ----------
    v_h_image_tuple : list of tuple[int, int, np.ndarray]
        A list of (v_index, h_index, grayscale image) tuples.

    Returns
    -------
    first_images_gpu : dict[int, cp.ndarray]
        Dictionary mapping v_index -> CuPy GPU array (image where h == 0).
    """
    first_images_gpu = {}

    for v, h, img in v_h_image_tuple:
        if h == 0 and v not in first_images_gpu:
            first_images_gpu[v] = cp.asarray(img)

    return first_images_gpu
    

def simple_stitch_gpu_2(images, overlap_pixels=0, overlap_side='right'):
    first_images_gpu = {
    v: cp.asarray(h_img_list[0][1])
    for v, h_img_list in images_by_v.items()
    }

    
import numpy as np

def get_overlaps(frameShift: int = 0, displacementShift: int = 0, image_list: list = []) -> list[int]:
    """
    Computes overlap values for each image index using a degree-3 polynomial model.
    """
    a = -1.5747863247863223
    b = 27.38986013986013
    c = -110.45843045843054
    d = 233.80419580419564

    overlaps = []
    for i in range(len(image_list)):
        x = i + frameShift
        y = a * x**3 + b * x**2 + c * x + d
        overlaps.append(int(round(y + displacementShift)))
    return overlaps

def get_overlaps_static(image_list: list, frameShift: int = 0, overlapShift: int = 0) -> list[int]:
    """
    Returns a shifted list of fixed overlap values, trimmed or padded with 0s to match the length of image_list.

    Parameters
    ----------
    image_list : list
        A list of any length; only its length is used.

    frameShift : int, optional
        Horizontal shift of the overlap list (positive = later values, negative = earlier).

    overlapShift : int, optional
        Vertical shift to be added to all overlap values.

    Returns
    -------
    list[int]
        A list of overlap values (with shift adjustments) of the same length as image_list.
    """

    center = 0
    always_zero = 0
    change_later = 0
    overlapShift = int(overlapShift)
    #base_overlaps = [always_zero, 0, 200, 200, 200, 200, 380, 300, 300, 300, 300,0, 0, 0, 0] #works
    #base_overlaps = [always_zero, 170, 210, 180, 180, 180, 400, 330, 300, 300, 300,170, 170, 170, 170]#works
    base_overlaps = [always_zero, 236, 230, 200, 200, 50, 236, 300, 290, 290, 290,236, 236, 236, 236]#works
    # Apply horizontal frame shift
    shifted = []
    for i in range(len(image_list)):
        idx = i + frameShift
        if 0 <= idx < len(base_overlaps):
            shifted.append(base_overlaps[idx] + overlapShift)
        else:
            shifted.append(0 + overlapShift)

    return shifted


import cv2
import numpy as np


def simple_stitch(
    images,
    frameShift=0,
    displacementShift=0,
    overlap_side='left',
    overlap_function=None,
    mark_boundaries=False
):
    """
    Simple horizontal stitching with dynamic overlap per image using a selectable model.
    Parameters
    ----------
    images : list of tuple[int, np.ndarray]
        List of (h_index, image) tuples.

    frameShift : int
        Shifts the input x-values or frame index in the overlap model.

    displacementShift : int
        Adds vertical bias to the overlap values.

    overlap_side : {'left', 'right'}
        Determines which side gets the overlap trimmed.

    overlap_function : callable, optional
        Function to compute overlaps. Must accept (image_list, frameShift, overlapShift).

    mark_boundaries : bool
        If True, blends overlap regions with alternating red/blue tint and transparency.

    Returns
    -------
    np.ndarray
        The final stitched panorama.
    """
    if not images:
        return None

    if overlap_function is None:
        overlap_function = get_overlaps_static

    overlaps = overlap_function(image_list=images, frameShift=frameShift, overlapShift=displacementShift)

    # Convert first image to BGR if needed
    panorama = images[0][1]
    if len(panorama.shape) == 2:
        panorama = cv2.cvtColor(panorama, cv2.COLOR_GRAY2BGR)

    for i, (_, img) in enumerate(images[1:], start=1):
        overlap = overlaps[i]

        # Convert new image to BGR if needed
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if mark_boundaries:
            alpha = 0.4  # transparency
            color = (0, 0, 255) if i % 2 == 1 else (255, 0, 0)  # Red or Blue
            color_overlay = np.full_like(img, color, dtype=np.uint8)
            img = cv2.addWeighted(img, 1 - alpha, color_overlay, alpha, 0)

        if overlap > 0:
            if overlap_side == 'right':
                panorama = np.hstack([panorama[:, :-overlap], img])
            else:  # 'left'
                panorama = np.hstack([panorama, img[:, overlap:]])
        else:
            panorama = np.hstack([panorama, img])

    return panorama
def stitch_images(
    image_paths_by_v: dict[int, list[Tuple[int, str]]],
    input_folder: str,
    output_folder: str,
    overlap_side: str = "left",
    frameShift: int = 0,
    displacementShift: int = 0,
    overlap_function=None,
    mark_boundaries: bool = False
):
    """
    Stitches horizontal image strips using the simple_stitch method, with optional overlap modeling.

    Parameters
    ----------
    image_paths_by_v : dict[int, list[Tuple[int, str]]]
        A dictionary where keys are vertical indices (v) and values are lists of
        tuples containing horizontal indices (h) and image file paths.

    input_folder : str
        The directory where the input images are located.

    output_folder : str
        The directory where stitched panoramas will be saved.

    overlap_side : {'left', 'right'}, optional
        Side of the image to trim during stitching overlap. Default is 'left'.

    frameShift : int, optional
        Horizontal index shift used for computing overlap. Default is 0.

    displacementShift : int, optional
        Vertical offset added to all computed overlap values. Default is 0.

    overlap_function : callable, optional
        Function to compute overlap values. Must accept arguments (image_list, frameShift, overlapShift).
        If None, defaults to `get_overlaps_static`.

    mark_boundaries : bool, optional
        If True, applies red/blue tint overlays on each overlap region for visualization.

    Returns
    -------
    None
        Saves each stitched horizontal image row (v-line) as a JPEG in `output_folder`.
    """
    os.makedirs(output_folder, exist_ok=True)

    if overlap_function is None:
        overlap_function = get_overlaps_static

    for v, items in image_paths_by_v.items():
        items.sort(key=lambda x: x[0])  # sort by h index

        images = []
        for h, path in items:
            full_path = os.path.join(input_folder, path)
            img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append((h, img))
            else:
                print(f"[Warning] Could not load image for v={v}, h={h}: {path}")

        if not images:
            print(f"[Skipped] No valid images found for v={v}")
            continue

        panorama = simple_stitch(
            images=images,
            frameShift=frameShift,
            displacementShift=displacementShift,
            overlap_side=overlap_side,
            overlap_function=overlap_function,
            mark_boundaries=mark_boundaries
        )

        if panorama is not None:
            output_path = os.path.join(output_folder, f"{v:02d}_stitched.jpg")
            cv2.imwrite(output_path, panorama)
            print(f"✔️ Saved stitched image for v={v} to {output_path}")
        else:
            print(f"[Error] Stitching failed for v={v}")

 
def test():
    test_dir = r"G:\pluma-2025-07-14 (2)\test"  # <- Adjust path if needed
    v_h_image_tuple = load_images_with_indices(test_dir)

    print(f"✅ Loaded {len(v_h_image_tuple)} image(s)")

    if v_h_image_tuple:
        # Show basic info
        for i, (v, h, img) in enumerate(v_h_image_tuple[:5]):  # just show first 5
            print(f"[{i}] v={v}, h={h}, shape={img.shape}, dtype={img.dtype}")

        # Optionally show image using OpenCV (press any key to continue)
        # cv2.imshow("First Image", images[0][2])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    else:
        print("⚠️ No images loaded. Check the directory and filename pattern.")
        
    #now test simple_stitch_gpu
    
def test2():
    """
    Test function to validate the full GPU-based stitching pipeline using a given folder.
    This function loads all images, extracts h=0 images to GPU, stitches by v,
    and saves output for multiple overlap pixel settings.
    """
    folder = r"G:\pluma-2025-07-14 (2)\test"  # <- change if needed
    overlap_pixels_list = list(range(0,512))
    overlap_side = "left"

    print("🚀 Running test2: Full GPU Stitch Pipeline with multiple overlap values")
    for overlap_pixels in overlap_pixels_list:
        print(f"\n🧪 Testing with overlap_pixels = {overlap_pixels}")
        simple_stitch_gpu(folder, overlap_pixels, overlap_side)
    print("\n✅ test2 completed.\n")


import os
import cv2

def test3(overlap_function=None):
    """
    Test function to validate the simple_stitch method using a selectable overlap model.
    By default, uses get_overlaps_static with frameShift = 0 and displacementShift = 0.
    It stitches each group of images with the same `v` value.

    Parameters
    ----------
    overlap_function : callable, optional
        The overlap function to use. Must accept image_list, frameShift, and overlapShift.
    """
    folder = r"G:\pluma-2025-07-14 (2)\test"
    frameShift = 0
    displacementShift = 0
    overlap_side = "left"

    if overlap_function is None:
        overlap_function = get_overlaps_static

    v_h_image_tuple = load_images_with_indices(folder)

    # Group by vertical index (v)
    images_by_v = {}
    for v, h, img in v_h_image_tuple:
        images_by_v.setdefault(v, []).append((h, img))

    for v, h_img_list in images_by_v.items():
        h_img_list.sort(key=lambda x: x[0])  # Sort by h index
        print(f"🧵 Stitching v={v} with {len(h_img_list)} images...")

        result = simple_stitch(
            images=h_img_list,
            frameShift=frameShift,
            displacementShift=displacementShift,
            overlap_side=overlap_side,
            overlap_function=overlap_function
        )

        output_path = os.path.join(folder, f"stitched_v{v:04d}_f{frameShift}_d{displacementShift}.jpg")
        cv2.imwrite(output_path, result)
        print(f"   ✔ Saved to {output_path}")
    
    print("\n✅ test3 completed.\n")


    



    
if __name__ == "__main__":
    #test2()
    test3()