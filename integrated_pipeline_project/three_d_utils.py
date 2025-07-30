import os
import numpy as np
from PIL import Image
import trimesh
from skimage import measure
import open3d as o3d
import re
Y_CUT_DISTANCE_MM = 2.5/512  #mm Ajusta este valor según tu aplicación
BLACK_THRESHOLD = 50
MM_PER_PIXEL = 2.3 / 512.0 
# ----------------------------------------
# Part 2: 3D Volume to Mesh
# ----------------------------------------


def load_and_build_volume(folder_path, black_threshold = BLACK_THRESHOLD ):
    """
    Convierte imágenes .tif blanco y negro en un volumen 3D binario.

    Input:
    - folder_path (str): ruta a la carpeta con imágenes .tif cuyos nombres comienzan con un número
    - black_threshold (float): umbral para clasificar píxeles como negros

    Output:
    - volume (np.ndarray): volumen binario (uint8)
    - slice_indices (list): índices extraídos del nombre del archivo
    - max_h (int): altura máxima
    - max_w (int): ancho máximo
    """
    files_with_idx = []
    filenames = os.listdir(folder_path)
    print(len(filenames), " files in ", folder_path)
    for fname in filenames:
        lower = fname.lower()
        if lower.endswith(".jpeg") or lower.endswith(".jpg"):
            match = re.match(r"^(\d+)", os.path.splitext(fname)[0])
            if match:
                idx = float(match.group(1))
                files_with_idx.append((idx, fname))

    if not files_with_idx:
        raise ValueError("No valid black-and-white JPG found with numeric filenames starting with digits.")

    files_with_idx.sort(key=lambda pair: pair[0])
    slice_indices = [pair[0] for pair in files_with_idx]
    sorted_filenames = [pair[1] for pair in files_with_idx]
    num_slices = len(sorted_filenames)

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
    volume = np.zeros((num_slices, max_h, max_w), dtype=np.uint8)

    for z_idx, arr in enumerate(loaded_images):
        h, w = arr.shape
        padded = np.full((max_h, max_w), 255, dtype=np.uint8)
        padded[0:h, 0:w] = arr
        black_ys, black_xs = np.where(padded <= black_threshold)
        volume[z_idx, black_ys, black_xs] = 1

    return volume, slice_indices, max_h, max_w


import numpy as np
import trimesh
from skimage import measure

def marching_cubes_to_trimesh_old(volume, slice_indices, image_height_px, image_width_px):
    """
    Genera una malla 3D (Trimesh) a partir de un volumen binario usando el algoritmo de Marching Cubes.

    Parámetros:
    - volume: volumen binario 3D con forma [num_slices, height, width] = [Y, Z, X]
    - slice_indices: índices que indican la posición de los cortes (Y) — no se usan directamente aquí
    - image_height_px: altura de cada imagen (en píxeles, eje Z real)
    - image_width_px: ancho de cada imagen (en píxeles, eje X real)

    Convenciones espaciales:
    - Eje X: columnas de la imagen → Ancho
    - Eje Z: filas de la imagen → Altura
    - Eje Y: dirección de apilamiento de las imágenes → Profundidad o posición entre cortes

    Escalado:
    - mm_per_pixel: resolución espacial en X y Z (mismo valor)
    - mm_per_slice: resolución en Y (distancia entre imágenes/slices)

    Retorna:
    - mesh_obj: objeto trimesh.Trimesh con coordenadas reales en milímetros
    """

    # Generar malla con marching cubes
    verts, faces, normals, values = measure.marching_cubes(volume, level=0.5, allow_degenerate=False)

    # Resoluciones espaciales
    mm_per_pixel = MM_PER_PIXEL     # milímetros por píxel en X y Z
    mm_per_slice = Y_CUT_DISTANCE_MM  # milímetros entre imágenes (Y)

    # Marching cubes devuelve vértices en orden: [Y, Z, X]
    y_index = verts[:, 0]  # índice del slice → eje Y
    z_index = verts[:, 1]  # índice vertical en la imagen → eje Z (altura)
    x_index = verts[:, 2]  # índice horizontal en la imagen → eje X (ancho)

    # Convertimos a coordenadas reales en milímetros
    x_mm = x_index * mm_per_pixel   # ancho real (X)
    y_mm = y_index * mm_per_slice   # profundidad real (Y)
    z_mm = z_index * mm_per_pixel   # altura real (Z)

    # Ensamblamos vértices en orden [X, Y, Z] con significado real
    verts_mm = np.column_stack([x_mm, y_mm, z_mm])

    # Crear la malla final
    mesh_obj = trimesh.Trimesh(vertices=verts_mm, faces=faces, process=False)

    return mesh_obj

import numpy as np
from skimage import measure
import trimesh

def marching_cubes_to_trimesh(volume, slice_indices, image_height_px, image_width_px):
    """
    Genera una malla 3D (Trimesh) a partir de un volumen binario usando el algoritmo de Marching Cubes.

    Parámetros:
    - volume: volumen binario 3D con forma [num_slices, height, width] = [Y, Z, X]
    - slice_indices: índices que indican la posición de los cortes (Y) — no se usan directamente aquí
    - image_height_px: altura de cada imagen (en píxeles, eje Z real)
    - image_width_px: ancho de cada imagen (en píxeles, eje X real)

    Convenciones espaciales:
    - Eje X: columnas de la imagen → Ancho
    - Eje Z: filas de la imagen → Altura
    - Eje Y: dirección de apilamiento de las imágenes → Profundidad o posición entre cortes

    Escalado:
    - mm_per_pixel: resolución espacial en X y Z (mismo valor)
    - mm_per_slice: resolución en Y (distancia entre imágenes/slices)

    Retorna:
    - mesh_obj: objeto trimesh.Trimesh con coordenadas reales en milímetros
    """
    print("[INFO] Volume shape:", volume.shape)

    # Recomendación 2: Convertir a float32 para ahorrar memoria
    if volume.dtype != np.float32:
        volume = volume.astype(np.float32)

    # Recomendación 5: Estimar el uso de memoria esperado
    step_size = 2
    estimated_faces = (np.array(volume.shape) // step_size).prod()
    estimated_bytes = estimated_faces * 3 * np.dtype(np.int32).itemsize
    estimated_gb = estimated_bytes / 1024**3
    print(f"[INFO] Estimación de memoria para 'faces': {estimated_gb:.2f} GiB")

    # Recomendación 1: Ejecutar marching_cubes con menor resolución (step_size)
    verts, faces, normals, values = measure.marching_cubes(
        volume,
        level=0.5,
        step_size=step_size,
        allow_degenerate=False
    )

    print(f"[INFO] Vértices generados: {len(verts)} - Caras: {len(faces)}")

    # Convertir a tipo más pequeño si es seguro
    if verts.shape[0] < 2**32:
        faces = faces.astype(np.uint32)

    # Resoluciones espaciales (constantes globales)
    mm_per_pixel = MM_PER_PIXEL       # milímetros por píxel en X y Z
    mm_per_slice = Y_CUT_DISTANCE_MM  # milímetros entre imágenes (Y)

    # Orden de vértices original: [Y, Z, X]
    y_index = verts[:, 0]
    z_index = verts[:, 1]
    x_index = verts[:, 2]

    # Convertir a coordenadas reales
    x_mm = x_index * mm_per_pixel
    y_mm = y_index * mm_per_slice
    z_mm = z_index * mm_per_pixel

    verts_mm = np.column_stack([x_mm, y_mm, z_mm])

    # Crear malla final
    mesh_obj = trimesh.Trimesh(vertices=verts_mm, faces=faces, process=False)

    return mesh_obj


# ----------------------------------------
# Part 3: Poisson Surface Reconstruction
# ----------------------------------------

def poisson_reconstruct_external(input_stl: str,
                                  output_stl: str,
                                  sample_count: int = 200000,
                                  depth: int = 9, radius = 0.3, max_nn=30, density_thresh = 0.03):
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
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn))
    pcd.orient_normals_consistent_tangent_plane(100)

    mesh_o3d, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)

    dens = np.asarray(densities)
    threshold = np.quantile(dens, density_thresh)
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
