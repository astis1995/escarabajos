
import os
import numpy as np
from PIL import Image
from skimage import measure
import trimesh
INTENSITY_THRESHOLD = 255 * 0.7
Y_CUT_DISTANCE_MM = 0.1  # distancia entre cortes panorámicos

def load_panorama_slices(folder_path):
    files_with_idx = []
    for fname in os.listdir(folder_path):
        if fname.lower().endswith("_stitched.jpg"):
            try:
                idx = int(fname.split("_")[0])  # v value
                files_with_idx.append((idx, fname))
            except ValueError:
                continue

    files_with_idx.sort()
    slice_indices = [idx for idx, _ in files_with_idx]
    sorted_filenames = [fname for _, fname in files_with_idx]
    num_slices = len(sorted_filenames)

    loaded_images = []
    heights = []
    widths = []

    for fname in sorted_filenames:
        img = Image.open(os.path.join(folder_path, fname)).convert("L")
        arr = np.array(img)
        heights.append(arr.shape[0])
        widths.append(arr.shape[1])
        loaded_images.append(arr)

    max_h = max(heights)
    max_w = max(widths)

    volume = np.zeros((num_slices, max_h, max_w), dtype=np.uint8)

    for z, arr in enumerate(loaded_images):
        h, w = arr.shape
        padded = np.full((max_h, max_w), 255, dtype=np.uint8)
        padded[0:h, 0:w] = arr
        black_ys, black_xs = np.where(padded <= INTENSITY_THRESHOLD)
        volume[z, black_ys, black_xs] = 1

    return volume, slice_indices, max_h, max_w

def marching_cubes_to_trimesh(volume, slice_indices, max_h, max_w):
    verts, faces, normals, values = measure.marching_cubes(volume, level=0.5)

    mm_per_pixel = 2.3 / 512.0
    mm_per_slice = Y_CUT_DISTANCE_MM

    z_idx = verts[:, 0]
    y_idx = verts[:, 1]
    x_idx = verts[:, 2]

    x_mm = x_idx * mm_per_pixel
    y_mm = z_idx * mm_per_slice
    z_mm = y_idx * mm_per_pixel

    verts_mm = np.column_stack([x_mm, y_mm, z_mm])

    mesh = trimesh.Trimesh(vertices=verts_mm, faces=faces, process=False)
    return mesh

from skimage import measure
from skimage.transform import resize
import trimesh
import numpy as np

def optimized_marching_cubes_to_trimesh(
    volume,
    mm_per_pixel=2.3 / 512.0,
    mm_per_slice=0.1,
    scale_factor=1.0,
    step_size=1,
    simplify_ratio=0.3
):
    """
    Generate a reduced 3D mesh from a binary volume.

    Parameters:
        volume: 3D numpy array (bool or uint8)
        mm_per_pixel: scale for x and z (default from your code)
        mm_per_slice: scale for y (between slices)
        scale_factor: 0.0–1.0 to downsample the volume
        step_size: int, voxel skipping in marching cubes (1 = full resolution)
        simplify_ratio: float 0.0–1.0 to reduce face count (0.3 = 70% smaller)

    Returns:
        trimesh.Trimesh object
    """

    # 1. Downsample if requested
    if scale_factor < 1.0:
        target_shape = np.round(np.array(volume.shape) * scale_factor).astype(int)
        volume = resize(
            volume,
            output_shape=target_shape,
            order=0,
            anti_aliasing=False,
            preserve_range=True
        ).astype(volume.dtype)

    # 2. Run marching cubes
    verts, faces, normals, values = measure.marching_cubes(volume, level=0.5, step_size=step_size)

    # 3. Convert voxel indices to mm space
    z_idx = verts[:, 0]  # slices
    y_idx = verts[:, 1]  # height
    x_idx = verts[:, 2]  # width

    x_mm = x_idx * mm_per_pixel
    y_mm = z_idx * mm_per_slice
    z_mm = y_idx * mm_per_pixel

    verts_mm = np.column_stack([x_mm, y_mm, z_mm]).astype(np.float32)

    mesh = trimesh.Trimesh(vertices=verts_mm, faces=faces, process=False)

    # 4. Optional simplification
    if simplify_ratio < 1.0 and len(mesh.faces) > 100:  # threshold for efficiency
        reduction_ratio = 1.0 - simplify_ratio  # ej. 0.7 para reducir a 30% original

        if hasattr(mesh, "simplify_quadric_decimation"):
            mesh = mesh.simplify_quadric_decimation(reduction_ratio)

        else:
            print("⚠️ Advertencia: Trimesh no soporta simplificación quadric. Continúa con malla original.")

    return mesh
