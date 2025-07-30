import tkinter as tk
import cv2
import os
import numpy as np
from typing import List, Tuple
import re

# Carpeta con imágenes
test_dir = r"G:\pluma-2025-07-14 (2)\test"

def load_images_with_indices(folder: str) -> List[Tuple[int, int, np.ndarray]]:
    v_h_image_tuple = []
    pattern = re.compile(r".*(\d+)-(\d+)_.*\.(tif|png|jpg)$", re.IGNORECASE)

    for filename in os.listdir(folder):
        match = pattern.match(filename)
        if match:
            v_index = int(match.group(1))
            h_index = int(match.group(2))
            path = os.path.join(folder, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                v_h_image_tuple.append((v_index, h_index, img))
    return v_h_image_tuple

def simple_stitch_with_manual_overlaps(images, overlaps, overlap_side='left'):
    if not images:
        return None

    panorama = images[0][1]
    if len(panorama.shape) == 2:
        panorama = cv2.cvtColor(panorama, cv2.COLOR_GRAY2BGR)

    for i, (_, img) in enumerate(images[1:], start=1):
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        overlap = overlaps[i] if i < len(overlaps) else 0
        alpha = 0.5
        color = (0, 0, 255) if i % 2 == 1 else (255, 0, 0)
        overlay = np.full_like(img, color, dtype=np.uint8)

        if overlap > 0:
            region_size = (512 - overlap) // 4
            if overlap_side == 'left':
                if region_size > 0:
                    img[:, overlap:overlap+region_size] = cv2.addWeighted(
                        img[:, overlap:overlap+region_size],
                        1 - alpha,
                        overlay[:, overlap:overlap+region_size],
                        alpha,
                        0
                    )
                    panorama[:, -region_size:] = cv2.addWeighted(
                        panorama[:, -region_size:],
                        1 - alpha,
                        overlay[:, :region_size],
                        alpha,
                        0
                    )
                img[:, overlap] = (0, 255, 0)  # Línea verde
            else:
                if region_size > 0:
                    panorama[:, -overlap-region_size:-overlap] = cv2.addWeighted(
                        panorama[:, -overlap-region_size:-overlap],
                        1 - alpha,
                        overlay[:, :region_size],
                        alpha,
                        0
                    )
                    img[:, :region_size] = cv2.addWeighted(
                        img[:, :region_size],
                        1 - alpha,
                        overlay[:, :region_size],
                        alpha,
                        0
                    )
                panorama[:, -overlap] = (0, 255, 0)  # Línea verde
        else:
            img[:, 0] = (0, 255, 0)

        if overlap > 0:
            if overlap_side == 'right':
                panorama = np.hstack([panorama[:, :-overlap], img])
            else:
                panorama = np.hstack([panorama, img[:, overlap:]])
        else:
            panorama = np.hstack([panorama, img])

    return panorama

# Cargar imágenes una vez
images_by_v = {}
images = load_images_with_indices(test_dir)
for v, h, img in images:
    images_by_v.setdefault(v, []).append((h, img))

first_v = min(images_by_v)
h_img_list = sorted(images_by_v[first_v], key=lambda x: x[0])

# GUI
root = tk.Tk()
root.title("Overlaps interactivos")

entry_frame = tk.Frame(root)
entry_frame.grid(row=0, column=0, padx=10, pady=10)

overlap_vars = []

def update_image(*args):
    try:
        overlaps = [int(var.get()) for var in overlap_vars]
    except ValueError:
        return
    stitched = simple_stitch_with_manual_overlaps(h_img_list, overlaps)
    if stitched is not None:
        cv2.imshow("Stitched Preview", stitched)
        cv2.waitKey(1)

for i in range(len(h_img_list)):
    var = tk.StringVar()
    var.set("0")
    var.trace_add("write", update_image)
    label = tk.Label(entry_frame, text=f"Overlap {i}")
    label.grid(row=i, column=0, sticky="e")
    entry = tk.Entry(entry_frame, textvariable=var, width=5)
    entry.grid(row=i, column=1, padx=2, pady=1)
    overlap_vars.append(var)

update_image()

root.mainloop()
cv2.destroyAllWindows()
