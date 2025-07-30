import os
import re
import cv2
import numpy as np

# --- Configuration ---
VERTICAL_DISPLACEMENT_PERCENTAGE = 0.15
MIN_SIMILARITY = 0.20

BASE_OVERLAP_RATIO = 0.30
MAX_OVERLAP_RATIO = 0.60
OVERLAP_STEP = 0.05

# --- Hardcoded folders ---
input_folder = r'E:\downloads\testfile_0-0-20250530-175517\originales\edges'
output_folder = r'E:\downloads\testfile_0-0-20250530-175517\originales\edges\outputpanorama'
iterations_folder = os.path.join(output_folder, 'iterations')
originals_folder = r'E:\downloads\testfile_0-0-20250530-175517\originales'
stitched_original_folder = os.path.join(originals_folder, 'stitched_original')

os.makedirs(output_folder, exist_ok=True)
os.makedirs(iterations_folder, exist_ok=True)
os.makedirs(stitched_original_folder, exist_ok=True)

filename_pattern = re.compile(r'testfile_(\d+)-(\d+)-\d+-\d+\.jpg')


def preprocess_image(img):
    return 255 - img  # invert colors


def vertical_offset_correlation(overlap_left, overlap_right, max_shift=10):
    h, w = overlap_left.shape
    best_corr = -1
    best_dy = 0

    for dy in range(-max_shift, max_shift + 1):
        if dy < 0:
            shifted_left = overlap_left[-dy:, :]
            shifted_right = overlap_right[:h + dy, :]
        elif dy > 0:
            shifted_left = overlap_left[:h - dy, :]
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
    h, w = img.shape
    if dy == 0:
        return img.copy()
    elif dy > 0:
        shifted = np.full_like(img, pad_value)
        shifted[dy:, :] = img[:h - dy, :]
        return shifted
    else:
        dy = -dy
        shifted = np.full_like(img, pad_value)
        shifted[:h - dy, :] = img[dy:, :]
        return shifted


def highlight_region(img, x_start, x_end, color):
    """Highlight vertical strip from x_start to x_end with given BGR color overlay."""
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    alpha = 0.4
    overlay = img_color.copy()
    overlay[:, x_start:x_end] = color
    cv2.addWeighted(overlay, alpha, img_color, 1 - alpha, 0, img_color)
    return img_color


def pad_to_width(img, target_width, pad_value=255):
    h, w = img.shape[:2]
    if w >= target_width:
        return img
    pad_width = target_width - w
    if img.ndim == 2:
        pad_array = np.full((h, pad_width), pad_value, dtype=img.dtype)
    else:
        pad_array = np.full((h, pad_width, img.shape[2]), pad_value, dtype=img.dtype)
    return np.hstack([img, pad_array])


def create_iteration_visualization(v, step, side,
                                   img_left, img_right,
                                   overlap_pixels, dy,
                                   panorama, added_region_start):
    red = (0, 0, 255)
    blue = (255, 0, 0)

    img_left_color = highlight_region(img_left, img_left.shape[1] - overlap_pixels, img_left.shape[1], red)
    img_right_color = highlight_region(img_right, 0, overlap_pixels, red)
    top_combined = np.hstack([img_left_color, img_right_color])

    panorama_color = cv2.cvtColor(panorama, cv2.COLOR_GRAY2BGR)

    if overlap_pixels > 0 and 0 <= added_region_start - overlap_pixels < panorama.shape[1]:
        x_overlap_start = added_region_start - overlap_pixels
        x_overlap_end = added_region_start
        panorama_color[:, x_overlap_start:x_overlap_end] = cv2.addWeighted(
            panorama_color[:, x_overlap_start:x_overlap_end], 0.6,
            np.full_like(panorama_color[:, x_overlap_start:x_overlap_end], red, dtype=np.uint8), 0.4, 0)

    if 0 <= added_region_start < panorama.shape[1]:
        x_added_start = added_region_start
        x_added_end = panorama.shape[1]
        panorama_color[:, x_added_start:x_added_end] = cv2.addWeighted(
            panorama_color[:, x_added_start:x_added_end], 0.6,
            np.full_like(panorama_color[:, x_added_start:x_added_end], blue, dtype=np.uint8), 0.4, 0)

    max_width = max(top_combined.shape[1], panorama_color.shape[1])

    top_combined_padded = pad_to_width(top_combined, max_width, pad_value=255)
    panorama_color_padded = pad_to_width(panorama_color, max_width, pad_value=255)

    spacing = 10
    spacer = 255 * np.ones((spacing, max_width, 3), dtype=np.uint8)

    combined_img = np.vstack([top_combined_padded, spacer, panorama_color_padded])

    filename = f"v{v}_{side}_step{step:03d}.jpg"
    path = os.path.join(iterations_folder, filename)
    cv2.imwrite(path, combined_img)
    print(f"Saved iteration visualization: {filename}")


def stitch_iterative(images, v,
                     base_overlap=BASE_OVERLAP_RATIO,
                     max_overlap=MAX_OVERLAP_RATIO,
                     overlap_step=OVERLAP_STEP,
                     min_similarity=MIN_SIMILARITY):
    n = len(images)
    center_idx = n // 2
    img_h, img_w = images[0].shape

    panorama = images[center_idx].copy()
    iteration_step = 0

    offsets = {center_idx: {'dy': 0, 'overlap': 0}}
    skipped_indices = set()

    create_iteration_visualization(
        v, iteration_step, "center",
        img_left=images[center_idx], img_right=images[center_idx],
        overlap_pixels=0, dy=0,
        panorama=panorama, added_region_start=panorama.shape[1]
    )
    iteration_step += 1

    overlap_ratio = base_overlap
    i = center_idx + 1
    while i < n:
        overlap_pixels = int(img_w * overlap_ratio)
        next_img = images[i]

        overlap_panorama = panorama[:, -overlap_pixels:]
        overlap_next = next_img[:, :overlap_pixels]

        max_shift = int(img_h * VERTICAL_DISPLACEMENT_PERCENTAGE)
        dy, corr = vertical_offset_correlation(overlap_panorama, overlap_next, max_shift=max_shift)

        print(f"Right stitching h={i}, overlap {overlap_ratio*100:.1f}%, similarity={corr:.4f}, vertical offset dy={dy} px")

        if corr < min_similarity:
            print(f"Skipping image h={i} due to low similarity ({corr:.4f})")
            skipped_indices.add(i)
            i += 1
            continue

        print(f"Shifting next image h={i} down by {dy} pixels before stitching")
        next_img_shifted = shift_image_vertically(next_img, -dy, pad_value=255)


        added_region_start = panorama.shape[1]
        panorama = np.hstack([panorama[:, :-overlap_pixels], next_img_shifted])

        offsets[i] = {'dy': dy, 'overlap': overlap_pixels}

        create_iteration_visualization(
            v, iteration_step, "right",
            img_left=images[i - 1], img_right=next_img,
            overlap_pixels=overlap_pixels, dy=dy,
            panorama=panorama, added_region_start=added_region_start
        )
        iteration_step += 1

        overlap_ratio = min(overlap_ratio + overlap_step, max_overlap)
        i += 1

    overlap_ratio = base_overlap
    i = center_idx - 1
    while i >= 0:
        overlap_pixels = int(img_w * overlap_ratio)
        prev_img = images[i]

        overlap_prev = prev_img[:, -overlap_pixels:]
        overlap_panorama = panorama[:, :overlap_pixels]

        max_shift = int(img_h * VERTICAL_DISPLACEMENT_PERCENTAGE)
        dy, corr = vertical_offset_correlation(overlap_prev, overlap_panorama, max_shift=max_shift)

        print(f"Left stitching h={i}, overlap {overlap_ratio*100:.1f}%, similarity={corr:.4f}, vertical offset dy={dy} px")

        if corr < min_similarity:
            print(f"Skipping image h={i} due to low similarity ({corr:.4f})")
            skipped_indices.add(i)
            i -= 1
            continue

        print(f"Shifting previous image h={i} down by {dy} pixels before stitching")
        prev_img_shifted = shift_image_vertically(prev_img, dy, pad_value=255)

        added_region_start = prev_img_shifted.shape[1]
        panorama = np.hstack([prev_img_shifted[:, :-overlap_pixels], panorama])

        offsets[i] = {'dy': dy, 'overlap': overlap_pixels}

        create_iteration_visualization(
            v, iteration_step, "left",
            img_left=prev_img, img_right=images[i + 1],
            overlap_pixels=overlap_pixels, dy=dy,
            panorama=panorama, added_region_start=added_region_start
        )
        iteration_step += 1

        overlap_ratio = min(overlap_ratio + overlap_step, max_overlap)
        i -= 1

    return panorama, offsets, skipped_indices


def apply_offsets_to_originals(original_images, offsets, skipped_indices):
    center_idx = len(original_images) // 2
    overlap_pixels = 0
    for v in offsets.values():
        if v['overlap'] > 0:
            overlap_pixels = v['overlap']
            break
    if overlap_pixels == 0:
        raise ValueError("No overlap info found in offsets.")

    panorama = original_images[center_idx].copy()

    i = center_idx + 1
    while i < len(original_images):
        if i in skipped_indices or i not in offsets:
            i += 1
            continue
        dy = offsets[i]['dy']
        overlap = offsets[i]['overlap']
        if overlap == 0:
            i += 1
            continue

        img = original_images[i]
        img_shifted = shift_image_vertically(img, dy, pad_value=0)

        panorama = np.hstack([panorama[:, :-overlap], img_shifted])
        i += 1

    i = center_idx - 1
    while i >= 0:
        if i in skipped_indices or i not in offsets:
            i -= 1
            continue
        dy = offsets[i]['dy']
        overlap = offsets[i]['overlap']
        if overlap == 0:
            i -= 1
            continue

        img = original_images[i]
        img_shifted = shift_image_vertically(img, dy, pad_value=0)

        panorama = np.hstack([img_shifted[:, :-overlap], panorama])
        i -= 1

    return panorama


# --- Main stitching process ---

image_groups = {}

for filename in os.listdir(input_folder):
    match = filename_pattern.match(filename)
    if match:
        h = int(match.group(1))
        v = int(match.group(2))
        path = os.path.join(input_folder, filename)
        image_groups.setdefault(v, []).append((h, path))

for v, image_list in image_groups.items():
    print(f"\nProcessing v={v} with {len(image_list)} images")

    image_list.sort(key=lambda x: x[0])

    images = []
    for h, path in image_list:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: could not load {path}")
            continue
        img = preprocess_image(img)
        images.append(img)
        print(f"Loaded and preprocessed h={h}")

    if not images:
        print(f"No valid images for v={v}, skipping")
        continue

    img_height = images[0].shape[0]
    if any(img.shape[0] != img_height for img in images):
        print(f"Error: Images have different heights for v={v}, skipping")
        continue

    panorama, offsets, skipped_indices = stitch_iterative(images, v=v)

    output_path = os.path.join(output_folder, f"{v}.jpg")
    cv2.imwrite(output_path, panorama)
    print(f"Saved panorama for v={v} to {output_path}")

    print(f"Applying saved offsets to original images for v={v}...")

    original_images = []
    for _, path in image_list:
        orig_path = os.path.join(originals_folder, os.path.basename(path))

        orig_img = cv2.imread(orig_path, cv2.IMREAD_GRAYSCALE)
        if orig_img is None:
            print(f"Warning: could not load original {orig_path}")
            original_images.append(None)
        else:
            original_images.append(orig_img)

    valid_idx_map = {}
    filtered_originals = []
    idx_new = 0
    for idx_old, img in enumerate(original_images):
        if img is not None:
            filtered_originals.append(img)
            valid_idx_map[idx_old] = idx_new
            idx_new += 1

    offsets_filtered = {}
    skipped_filtered = set()
    for idx_old, val in offsets.items():
        if idx_old in valid_idx_map:
            offsets_filtered[valid_idx_map[idx_old]] = val
    for idx_old in skipped_indices:
        if idx_old in valid_idx_map:
            skipped_filtered.add(valid_idx_map[idx_old])

    stitched_orig = apply_offsets_to_originals(filtered_originals, offsets_filtered, skipped_filtered)

    out_orig_path = os.path.join(stitched_original_folder, f"{v}_original.jpg")
    cv2.imwrite(out_orig_path, stitched_orig)
    print(f"Saved stitched original panorama for v={v} to {out_orig_path}")

print("\nAll done.")
