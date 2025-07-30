import os
import numpy as np
from PIL import Image
import trimesh
from skimage import measure
import open3d as o3d

Y_CUT_DISTANCE_MM = 0.23  # Ajusta este valor según tu aplicación

# ----------------------------------------
# Part 2: 3D Volume to Mesh
# ----------------------------------------

def load_and_build_volume(folder_path, black_threshold=255/2):
    """
    Convierte imágenes .jpg blanco y negro en un volumen 3D binario.

    Input:
    - folder_path (str): ruta a la carpeta con imágenes .jpg indexadas por número
    - black_threshold (float): umbral para clasificar píxeles como negros

    Output:
    - volume (np.ndarray): volumen binario (uint8)
    - slice_indices (list): índices de las imágenes
    - max_h (int): altura máxima
    - max_w (int): ancho máximo
    """
    files_with_idx = []
    for fname in os.listdir(folder_path):
        lower = fname.lower()
        if lower.endswith(".jpg") or lower.endswith(".jpeg"):
            name_no_ext = os.path.splitext(fname)[0]
            try:
                idx = int(name_no_ext)
            except ValueError:
                continue
            files_with_idx.append((idx, fname))

    if not files_with_idx:
        raise ValueError("No valid black-and-white JPGs found with numeric names.")

    files_with_idx.sort(key=lambda pair: pair[0])
    slice_indices = [pair[0] for pair in files_with_idx]
    sorted_filenames = [pair[1] for pair in files_with_idx]
    num_slices = len(sorted_filenames)

    loaded_images = []
    heights = []
    widths = []
    for fname in sorted_filenames:
        #convert to luminance
        arr = np.array(Image.open(os.path.join(folder_path, fname)).convert("L"))
        #add haight and width info
        heights.append(arr.shape[0])
        widths.append(arr.shape[1])
        #append image
        loaded_images.append(arr)

    max_h = max(heights)
    max_w = max(widths)
    volume = np.zeros((num_slices, max_h, max_w), dtype=np.uint8)

    for z_idx, arr in enumerate(loaded_images):
        h, w = arr.shape
        padded = np.full((max_h, max_w), 255, dtype=np.uint8)
        padded[0:h, 0:w] = arr
        black_ys, black_xs = np.where(padded <= black_threshold)
        volume[z_idx, black_ys, black_xs] = 1

    return volume, slice_indices, max_h, max_w


def marching_cubes_to_trimesh(volume, slice_indices, max_h, max_w):
    """
    Genera una malla Trimesh desde un volumen binario.

    Input:
    - volume: volumen 3D binario
    - slice_indices: índices de corte (Z)
    - max_h, max_w: altura y ancho máximos

    Output:
    - mesh_obj (trimesh.Trimesh): malla reconstruida
    """
    verts, faces, normals, values = measure.marching_cubes(volume, level=0.5, allow_degenerate=False)

    mm_per_pixel = 2.3 / 512.0
    mm_per_slice = Y_CUT_DISTANCE_MM

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
# Part 3: Poisson Surface Reconstruction
# ----------------------------------------

def poisson_reconstruct_external(input_stl: str,
                                  output_stl: str,
                                  sample_count: int = 200000,
                                  depth: int = 9):
    """
    Aplica reconstrucción Poisson usando Open3D.

    Input:
    - input_stl: ruta a STL de entrada
    - output_stl: ruta para guardar el STL final
    - sample_count: número de puntos a muestrear
    - depth: profundidad del árbol octree

    Output:
    - None (guarda un archivo STL final)
    """
    tri = trimesh.load(input_stl)
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
    print(f"✅ Guardado STL Poisson en: {output_stl}")
