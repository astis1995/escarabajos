import os
import numpy as np
import trimesh
from PIL import Image
from skimage.measure import marching_cubes
import open3d as o3d
import re
Y_CUT_DISTANCE_MM = 0.23

# ----------------------------------------
# Part 1: Load volume from image slices
# ----------------------------------------

def load_and_build_volume(folder_path, black_threshold=127.5):
    files_with_idx = []
    list_dir = os.listdir(folder_path)
    if not list_dir:
        raise ValueError("Empty folder, please check")
    for fname in list_dir:
        print("First 10", list_dir[:10])
        lower = fname.lower()

        # Only process if it's an image file
        if lower.endswith(".jpg") or lower.endswith(".jpeg"):
            # Use regex to extract leading numeric index
            match = re.match(r'^(\d+(?:\.\d+)?)', fname)
            if match:
                try:
                    idx = float(match.group(1))  # Convert to float if needed
                    files_with_idx.append((idx, fname))
                except ValueError:
                    continue

    if not files_with_idx:
        raise ValueError("No valid black-and-white JPGs found with numeric names.")

    files_with_idx.sort(key=lambda pair: pair[0])
    slice_indices = [pair[0] for pair in files_with_idx]
    sorted_filenames = [pair[1] for pair in files_with_idx]

    loaded_images, heights, widths = [], [], []
    for fname in sorted_filenames:
        arr = np.array(Image.open(os.path.join(folder_path, fname)).convert("L"))
        heights.append(arr.shape[0])
        widths.append(arr.shape[1])
        loaded_images.append(arr)

    max_h, max_w = max(heights), max(widths)
    num_slices = len(sorted_filenames)
    volume = np.zeros((num_slices, max_h, max_w), dtype=np.uint8)

    for z_idx, arr in enumerate(loaded_images):
        h, w = arr.shape
        padded = np.full((max_h, max_w), 255, dtype=np.uint8)
        padded[:h, :w] = arr
        black_ys, black_xs = np.where(padded <= black_threshold)
        volume[z_idx, black_ys, black_xs] = 1

    return volume, slice_indices, max_h, max_w

# ----------------------------------------
# Part 2: Chunked Marching Cubes
# ----------------------------------------

def marching_cubes_chunked_old(volume, chunk_bytes_limit=10 * 1024**3):
    chunk_voxels = chunk_bytes_limit // 1  # uint8 = 1 byte
    z, y, x = volume.shape
    chunk_depth = max(1, chunk_voxels // (y * x))

    verts_all = []
    faces_all = []
    vert_offset = 0

    for z_start in range(0, z, chunk_depth):
        z_end = min(z_start + chunk_depth + 1, z)
        chunk = volume[z_start:z_end]

        if np.sum(chunk) == 0:
            continue

        try:
            verts, faces, _, _ = marching_cubes(chunk, level=0.5)
        except ValueError:
            continue

        verts[:, 0] += z_start
        verts_all.append(verts)
        faces_all.append(faces + vert_offset)
        vert_offset += len(verts)

    verts_all = np.concatenate(verts_all)
    faces_all = np.concatenate(faces_all)

    mm_per_pixel = 2.3 / 512.0
    mm_per_slice = Y_CUT_DISTANCE_MM

    z_idx, y_idx, x_idx = verts_all[:, 0], verts_all[:, 1], verts_all[:, 2]
    x_mm = x_idx * mm_per_pixel
    y_mm = z_idx * mm_per_slice
    z_mm = y_idx * mm_per_pixel

    verts_mm = np.column_stack([x_mm, y_mm, z_mm]).astype(np.float32)
    return trimesh.Trimesh(vertices=verts_mm, faces=faces_all, process=False)

import numpy as np
from skimage import measure
import trimesh

def marching_cubes_to_trimesh(volume, slice_indices=None, max_h=None, max_w=None, step_size=2):
    """
    Applies marching cubes to a 3D volume and returns a Trimesh object.

    Parameters:
    - volume: 3D numpy array (binary or grayscale)
    - slice_indices, max_h, max_w: Optional metadata (unused here but could be for trimming)
    - step_size: Controls mesh resolution. Higher = coarser, faster, lower memory.

    Returns:
    - mesh: trimesh.Trimesh object
    """
    print("[INFO] Volume shape:", volume.shape)

    # --- Step 5: Estimate memory before running
    estimated_faces = (np.array(volume.shape) // step_size).prod()
    estimated_bytes = estimated_faces * 3 * np.dtype(np.int32).itemsize
    estimated_gb = estimated_bytes / 1024**3
    print(f"[INFO] Estimated memory for faces: {estimated_gb:.2f} GiB")

    # --- Step 2: Ensure dtype is float32
    if volume.dtype != np.float32:
        volume = volume.astype(np.float32)

    # --- Step 1: Apply marching cubes with reduced step_size
    print("[INFO] Running marching_cubes...")
    verts, faces, normals, values = measure.marching_cubes(
        volume,
        level=0.5,
        step_size=step_size,
        allow_degenerate=False
    )

    print(f"[INFO] Mesh created with {len(verts)} vertices and {len(faces)} faces.")

    # Optional: reduce face dtype size
    if verts.shape[0] < 2**32:
        faces = faces.astype(np.uint32)

    # Create the mesh
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=False)
    return mesh

# ----------------------------------------
# Part 3: Poisson Surface Reconstruction
# ----------------------------------------

def poisson_reconstruct_external(input_stl: str, output_stl: str, sample_count=200000, depth=9):
    tri = trimesh.load(input_stl, process=False)
    if not isinstance(tri, trimesh.Trimesh):
        raise ValueError(f"'{input_stl}' did not load as a single Trimesh.")

    points, _ = trimesh.sample.sample_surface(tri, sample_count)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(100)

    mesh_o3d, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    dens = np.asarray(densities)
    threshold = np.quantile(dens, 0.10)
    keep_vert_indices = np.where(dens > threshold)[0]
    mesh_o3d = mesh_o3d.select_by_index(keep_vert_indices)
    mesh_o3d.remove_unreferenced_vertices()

    triangle_clusters, cluster_n_triangles, _ = mesh_o3d.cluster_connected_triangles()
    labels = np.asarray(triangle_clusters)
    largest_cluster = np.argmax(cluster_n_triangles)
    triangles_to_remove = np.where(labels != largest_cluster)[0]
    mesh_o3d.remove_triangles_by_index(triangles_to_remove)
    mesh_o3d.remove_unreferenced_vertices()

    mesh_o3d.compute_vertex_normals()

    success = o3d.io.write_triangle_mesh(output_stl, mesh_o3d, write_ascii=False)
    if not success:
        raise RuntimeError("Failed to write STL. Ensure mesh has valid normals.")
    print(f"Saved Poisson‐reconstructed external surface to '{output_stl}'")
