import os
import cv2
import numpy as np
from image_utilities import extract_identifiers  # Assumes you extract (v, v)


def vertical_offset_correlation(overlap_left, overlap_right, max_shift=10):
    v, w = overlap_left.shape
    best_corr = -1
    best_dy = 0
    for dy in range(-max_shift, max_shift + 1):
        if dy < 0:
            shifted_left = overlap_left[-dy:, :]
            shifted_right = overlap_right[:v + dy, :]
        elif dy > 0:
            shifted_left = overlap_left[:v - dy, :]
            shifted_right = overlap_right[dy:, :]
        else:
            shifted_left = overlap_left
            shifted_right = overlap_right

        if shifted_left.size == 0 or shifted_right.size == 0:
            continue

        left_norm = (shifted_left - shifted_left.mean()) / (shifted_left.std() + 1e-8)
        right_norm = (shifted_right - shifted_right.mean()) / (shifted_right.std() + 1e-8)
        corr = np.mean(left_norm * right_norm)

        if corr > best_corr:
            best_corr = corr
            best_dy = dy
    return best_dy, best_corr

def shift_image_vertically(img, dy, pad_value=255):
    v, w = img.shape
    shifted = np.full_like(img, pad_value)
    if dy > 0:
        shifted[dy:] = img[:v - dy]
    elif dy < 0:
        shifted[:v + dy] = img[-dy:]
    else:
        shifted = img.copy()
    return shifted

def get_overlaps(frameShift: int = 0, displacementShift: int = 0, image_list: list[int] = []) -> list[int]:
    """
    Computes overlap values for each image index using a degree-3 polynomial model.

    Parameters
    ----------
    frameShift : int
        Horizontal shift applied to the x values before evaluating the polynomial.

    displacementShift : int
        Vertical shift added to the resulting overlap values.

    image_list : list
        A list of items (e.g., image filenames or frames). Only its length is used.

    Returns
    -------
    list[int]
        List of overlap values (integers), same length as image_list.
    """

    # Coefficients of the degree-3 polynomial
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


def simple_stitch(images, overlap_pixels=0, overlap_side='right'):
    """
    Simple horizontal stitching with optional overlap.
    overlap_side: 'left' = new image overlaps left image
                  'right' = left image overlaps new image
    """
    if not images:
        return None
    
    overlaps = [181, 181,181, 208, 0, 0, 305, 315, 351, 277, 279]
    panorama = images[0][1]  # first image (already preprocessed)
    for _, img in images[1:]:
        if overlap_pixels > 0:
            if overlap_side == 'right':
                panorama = np.hstack([panorama[:, :-overlap_pixels], img])
            else:  # 'left'
                panorama = np.hstack([panorama, img[:, overlap_pixels:]])
        else:
            panorama = np.hstack([panorama, img])
    return panorama

def stitch_center(images, base_overlap, center_offset=0, skew=0):
    """
    Stitch images horizontally starting from a shifted center image.
    Overlap increases linearly, with optional skew on the left side.

    Parameters:
        images: list of (index, np.array) tuples (must be sorted by index)
        base_overlap: int, base pixel overlap per image
        center_offset: int, shift from the default center index
        skew: int, adds to base_overlap on the left side (asymmetrical growth)

    Returns:
        panorama: np.array stitched image
    """
    #print("images", images)
    if not images:
        return None

    num = len(images)

    # Compute center index with offset
    base_center = num // 2 if num % 2 == 1 else (num // 2 - 1)
    center_idx = base_center + center_offset

    if not (0 <= center_idx < num):
        raise ValueError(f"Invalid center index {center_idx} after applying offset {center_offset}")

    # Set center image
    panorama = images[center_idx][1].copy()
    left_images = images[:center_idx][::-1]    # from center-1 down to 0
    right_images = images[center_idx + 1:]     # from center+1 up

    # ➤ Stitch to the right (normal overlap)
    for i, (_, img) in enumerate(right_images, 1):
        overlap = i * base_overlap
        panorama = np.hstack([
            panorama[:, :-overlap],
            img
        ])

    # ➤ Stitch to the left (skewed overlap)
    for i, (_, img) in enumerate(left_images, 1):
        left_overlap = i * (base_overlap + skew)
        panorama = np.hstack([
            img[:, :-left_overlap],
            panorama
        ])

    return panorama




def advanced_stitch(images, vertical_displacement_percentage=0.15, min_similarity=0.2):
    """
    Aligns images using correlation and stitches them with vertical adjustment.
    """
    if not images:
        return None

    center_idx = len(images) // 2
    panorama = images[center_idx][1].copy()
    img_h, img_w = panorama.shape

    for direction in [-1, 1]:
        i = center_idx
        while 0 <= i + direction < len(images):
            left = panorama if direction == 1 else images[i + direction][1]
            right = images[i + direction][1] if direction == 1 else panorama

            overlap_pixels = int(img_w * 0.05)
            max_shift = int(img_h * vertical_displacement_percentage)

            overlap_left = left[:, -overlap_pixels:]
            overlap_right = right[:, :overlap_pixels]

            dy, corr = vertical_offset_correlation(overlap_left, overlap_right, max_shift)

            if corr < min_similarity:
                break

            shifted = shift_image_vertically(right, -dy if direction == 1 else dy)

            if direction == 1:
                panorama = np.hstack([panorama[:, :-overlap_pixels], shifted])
            else:
                panorama = np.hstack([shifted[:, :-overlap_pixels], panorama])

            i += direction
    return panorama


def stitch_images(
    image_paths_by_v,
    input_folder, 
    output_folder,
    method="stitch_center",
    overlap_pixels=300,
    overlap_side="left",
    vertical_displacement_percentage=0.0,
    min_similarity=0.2,
    center_offset=-(15/2-3),
    skew=5
):
    """
    Stitches each horizontal group using the selected method.
    method: 'simple_stitch' or 'advanced_stitch'
    """
    #print("Performing stitch images for ", image_paths_by_v.items())
    os.makedirs(output_folder, exist_ok=True)

    for v, items in image_paths_by_v.items():
        items.sort(key=lambda x: x[0])  # sort by h index
        #print("items",items)
        images = []
        for h, path in items:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append((h, img))

        if not images:
            continue

        if method == "simple_stitch":
            panorama = simple_stitch(images, overlap_pixels, overlap_side)
        elif method == "advanced_stitch":
            panorama = advanced_stitch(images, vertical_displacement_percentage, min_similarity)
        elif method == "stitch_center":
            panorama = stitch_center(
                images,
                base_overlap=overlap_pixels,
                center_offset=center_offset,
                skew=skew
            )

        else:
            raise ValueError(f"Unknown method: {method}")

        if panorama is not None:
            output_path = os.path.join(output_folder, f"{v:02d}_stitched.jpg")
            cv2.imwrite(output_path, panorama)
            print(f"Saved {method} panorama v={v:02d} to {output_path}")