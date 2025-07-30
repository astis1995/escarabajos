import os
import re
import cv2
import numpy as np
from PIL import Image
from skimage import measure
import trimesh
import open3d as o3d

# ----------------------------------------
# Part 1: Panorama Stitching (from blending9.py)
# ----------------------------------------

# --- Configuration ---
VERTICAL_DISPLACEMENT_PERCENTAGE = 0.15
MIN_SIMILARITY = 0.20

BASE_OVERLAP_RATIO = (2.05-2)/2.05
MAX_OVERLAP_RATIO = 0.15
OVERLAP_STEP = 0.05

# --- Hardcoded folders (adjust as needed) ---
input_folder = r'C:\Users\Labo402\client_received\abejon4'
output_folder = r'C:\Users\Labo402\client_received\abejon4\outputpanorama'
iterations_folder = os.path.join(output_folder, 'iterations')
originals_folder = r'C:\Users\Labo402\client_received\abejon4'
stitched_original_folder = os.path.join(originals_folder, 'stitched_original')

os.makedirs(output_folder, exist_ok=True)
os.makedirs(iterations_folder, exist_ok=True)
os.makedirs(stitched_original_folder, exist_ok=True)

filename_pattern = re.compile(r'.*_(\d+)-(\d+)-\d+-\d+\.jpg')


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


# ----------------------------------------
# Part 2: 3D Volume to Mesh (from three_d_model2.py)
# ----------------------------------------

def load_and_build_volume(folder_path):
    """
    1) Finds all black-&-white JPEGs in folder whose names parse as numeric slice indices.
    2) Sorts them by their numeric index.
    3) Pads each image to the max width/height across the folder.
    4) Builds a 3D NumPy volume of shape (num_slices, max_h, max_w), dtype=uint8:
         volume[z, y, x] = 1 if pixel at (x,y) in slice z was black, else 0.
    5) Returns: volume (uint8), slice_indices (sorted list of float), max_h, max_w
    """
    files_with_idx = []
    for fname in os.listdir(folder_path):
        lower = fname.lower()
        if lower.endswith(".jpg") or lower.endswith(".jpeg"):
            name_no_ext = os.path.splitext(fname)[0]
            try:
                idx = float(name_no_ext)
            except ValueError:
                continue
            files_with_idx.append((idx, fname))

    if not files_with_idx:
        raise ValueError("No valid black-and-white JPGs found with numeric names.")

    # Sort by numeric slice‐index
    files_with_idx.sort(key=lambda pair: pair[0])
    slice_indices = [pair[0] for pair in files_with_idx]
    sorted_filenames = [pair[1] for pair in files_with_idx]
    num_slices = len(sorted_filenames)

    # Load each image as grayscale, collect sizes
    loaded_images = []
    heights = []
    widths = []
    for fname in sorted_filenames:
        arr = np.array(Image.open(os.path.join(folder_path, fname)).convert("L"))
        heights.append(arr.shape[0])
        widths.append(arr.shape[1])
        loaded_images.append(arr)

    max_h = max(heights)
    max_w = max(widths)

    # Initialize a volume of zeros (uint8) and fill black pixels with 1
    volume = np.zeros((num_slices, max_h, max_w), dtype=np.uint8)

    for z_idx, arr in enumerate(loaded_images):
        h, w = arr.shape
        # Pad on bottom & right so shape = (max_h, max_w)
        if h < max_h or w < max_w:
            padded = np.full((max_h, max_w), 255, dtype=np.uint8)
            padded[0:h, 0:w] = arr
        else:
            padded = arr

        black_ys, black_xs = np.where(padded == 0)
        volume[z_idx, black_ys, black_xs] = 1

    return volume, slice_indices, max_h, max_w


def marching_cubes_to_trimesh(volume, slice_indices, max_h, max_w):
    """
    1) Runs marching_cubes on the 3D volume to get verts, faces in voxel‐space.
    2) Reorders axes and scales to mm:
       x_mm = x_idx * (2.3 / 512)
       y_mm = z_idx * 0.23
       z_mm = y_idx * (2.3 / 512)
    3) Returns a trimesh.Trimesh mesh.
    """
    verts, faces, normals, values = measure.marching_cubes(volume, level=0.5, allow_degenerate=False)

    mm_per_pixel = 2.3 / 512.0
    mm_per_slice = 0.23

    z_idx = verts[:, 0]
    y_idx = verts[:, 1]
    x_idx = verts[:, 2]

    x_mm = x_idx * mm_per_pixel
    y_mm = z_idx * mm_per_slice
    z_mm = y_idx * mm_per_pixel

    verts_mm = np.column_stack([x_mm, y_mm, z_mm])

    mesh_obj = trimesh.Trimesh(vertices=verts_mm, faces=faces, process=False)
    return mesh_obj


# ----------------------------------------
# Part 3: External Poisson Reconstruction (from external_surface.py)
# ----------------------------------------

def poisson_reconstruct_external(input_stl: str,
                                  output_stl: str,
                                  sample_count: int = 200000,
                                  depth: int = 9):
    """
    1) Load the input STL via trimesh.
    2) Uniformly sample 'sample_count' points from its surface.
    3) Convert those points to an Open3D PointCloud, estimate normals.
    4) Run Poisson surface reconstruction (with given 'depth').
    5) Remove low-density triangles to eliminate interior geometry.
    6) Keep only the largest connected component.
    7) Compute normals on the final mesh.
    8) Save the resulting external surface as output_stl.
    """
    # Step 1: Load with trimesh
    tri = trimesh.load(input_stl)
    if not isinstance(tri, trimesh.Trimesh):
        raise ValueError(f"'{input_stl}' did not load as a single Trimesh.")

    # Step 2: Sample points uniformly from the mesh surface
    points, _ = trimesh.sample.sample_surface(tri, sample_count)

    # Step 3: Build Open3D PointCloud & estimate normals
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.1, max_nn=30
    ))
    pcd.orient_normals_consistent_tangent_plane(100)

    # Step 4: Poisson reconstruction
    mesh_o3d, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth
    )

    # Step 5: Remove low-density triangles (interior)
    dens = np.asarray(densities)
    threshold = np.quantile(dens, 0.10)  # discard bottom 10% density
    keep_vert_indices = np.where(dens > threshold)[0]
    mesh_o3d = mesh_o3d.select_by_index(keep_vert_indices)
    mesh_o3d.remove_unreferenced_vertices()

    # Step 6: Keep only largest connected component
    triangle_clusters, cluster_n_triangles, _ = mesh_o3d.cluster_connected_triangles()
    labels = np.asarray(triangle_clusters)
    largest_cluster = np.argmax(cluster_n_triangles)
    triangles_to_remove = np.where(labels != largest_cluster)[0]
    mesh_o3d.remove_triangles_by_index(triangles_to_remove)
    mesh_o3d.remove_unreferenced_vertices()

    # Step 7: Compute normals on the final mesh
    mesh_o3d.compute_vertex_normals()

    # Step 8: Save to STL
    success = o3d.io.write_triangle_mesh(output_stl, mesh_o3d, write_ascii=False)
    if not success:
        raise RuntimeError("Failed to write STL. Ensure mesh has valid normals.")
    print(f"Saved Poisson‐reconstructed external surface to '{output_stl}'")


# ----------------------------------------
# Main Pipeline
# ----------------------------------------

if __name__ == "__main__":
    # --- Step A: Run the panorama stitching ---
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

    print("\nPanorama stitching complete.")

    # --- Step B: Generate a marching-cubes mesh from the panoramas ---
    print("\nBuilding 3D volume from panorama slices...")
    volume, slice_indices, max_h, max_w = load_and_build_volume(output_folder)
    print("Volume built. Running marching cubes...")

    mesh_3d = marching_cubes_to_trimesh(volume, slice_indices, max_h, max_w)
    mesh_output_file = "marching_cubes_mesh.stl"
    mesh_3d.export(mesh_output_file)
    print(f"Saved single‐surface mesh to '{mesh_output_file}'")

    # --- Step C: Poisson-based external surface reconstruction ---
    print("\nRunning Poisson reconstruction to extract external surface...")
    external_output_file = "external_surface.stl"
    poisson_reconstruct_external(mesh_output_file, external_output_file, sample_count=200000, depth=9)

    print("\nAll steps completed successfully.")
