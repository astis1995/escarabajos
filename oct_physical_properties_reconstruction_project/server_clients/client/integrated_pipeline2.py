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
X_WIDTH = 2.075
Y_DISPLACEMENT = 0.05

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


def stitch_iterative(images, v, **kwargs):
    panorama = images[0].copy()
    for img in images[1:]:
        panorama = np.hstack([panorama, img])
    return panorama, {}, set()



def apply_offsets_to_originals(original_images, offsets, skipped_indices):
    panorama = original_images[0].copy()
    for img in original_images[1:]:
        if img is not None:
            panorama = np.hstack([panorama, img])
    return panorama



# ----------------------------------------
# Part 2: 3D Volume to Mesh (from three_d_model2.py)
# ----------------------------------------

def load_and_build_volume(folder_path):
    """
    1) Carga imágenes JPG de la carpeta ordenadas por nombre.
    2) Las convierte en volumen 3D binario: 0 si blanco, 1 si no blanco.
    3) Recorta el 5% superior en eje Y.
    4) Devuelve: volume (uint8), max_h, max_w
    """
    filenames = sorted(
        [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg"))]
    )
    if not filenames:
        raise ValueError("No se encontraron imágenes .jpg en el folder.")

    loaded_images = []
    for fname in filenames:
        path = os.path.join(folder_path, fname)
        arr = np.array(Image.open(path).convert("L"))
        binary = (arr < 255).astype(np.uint8)  # 1 si no es blanco, 0 si blanco
        loaded_images.append(binary)

    volume = np.stack(loaded_images, axis=0)

    # Eliminar el 5% superior (en eje Y)
    cut_top = int(volume.shape[1] * 0.05)
    volume = volume[:, cut_top:, :]

    return volume, volume.shape[1], volume.shape[2]



def marching_cubes_to_trimesh(volume, max_h, max_w):
    """
    Corre marching cubes sobre volumen binario (ya recortado).
    Escala el volumen a milímetros.
    """
    verts, faces, normals, _ = measure.marching_cubes(volume, level=0.5)

    mm_per_pixel = X_WIDTH / 512.0
    mm_per_slice = Y_DISPLACEMENT

    z_idx = verts[:, 0]  # profundidad
    y_idx = verts[:, 1]  # vertical
    x_idx = verts[:, 2]  # horizontal

    x_mm = x_idx * mm_per_pixel
    y_mm = z_idx * mm_per_slice
    z_mm = y_idx * mm_per_pixel

    verts_mm = np.column_stack([x_mm, y_mm, z_mm])

    mesh = trimesh.Trimesh(vertices=verts_mm, faces=faces, process=False)
    return mesh



# ----------------------------------------
# Part 3: External Poisson Reconstruction (from external_surface.py)
# ----------------------------------------

def poisson_reconstruct_external(input_stl: str,
                                  output_stl: str,
                                  sample_count: int = 200000,
                                  depth: int = 9):
    """
    Corre Poisson surface reconstruction sin eliminar partes interiores.
    """
    tri = trimesh.load(input_stl)
    if not isinstance(tri, trimesh.Trimesh):
        raise ValueError(f"'{input_stl}' no es un Trimesh válido.")

    points, _ = trimesh.sample.sample_surface(tri, sample_count)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.1, max_nn=30
    ))
    pcd.orient_normals_consistent_tangent_plane(100)

    mesh_o3d, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth
    )
    mesh_o3d.compute_vertex_normals()

    o3d.io.write_triangle_mesh(output_stl, mesh_o3d, write_ascii=False)
    print(f"✅ STL guardado en '{output_stl}'")



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

        panorama, offsets, skipped_indices = stitch_iterative(images, v)


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
    volume, max_h, max_w = load_and_build_volume(output_folder)


    print("Volume built. Running marching cubes...")

    mesh_3d = marching_cubes_to_trimesh(volume, max_h, max_w)
    mesh_output_file = "marching_cubes_mesh.stl"
    mesh_3d.export(mesh_output_file)
    print(f"Saved single‐surface mesh to '{mesh_output_file}'")

    # --- Step C: Poisson-based external surface reconstruction ---
    print("\nRunning Poisson reconstruction to extract external surface...")
    external_output_file = "external_surface.stl"
    poisson_reconstruct_external(mesh_output_file, external_output_file, sample_count=200000, depth=9)

    print("\nAll steps completed successfully.")
