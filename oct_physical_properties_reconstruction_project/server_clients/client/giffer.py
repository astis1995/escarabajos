import os
import imageio
import numpy as np

def pad_image_to_size(img, target_height, target_width, pad_value=255):
    h, w = img.shape[:2]
    pad_bottom = target_height - h
    pad_right = target_width - w
    if pad_bottom < 0 or pad_right < 0:
        raise ValueError("Target size must be greater or equal to image size")
    
    if img.ndim == 2:
        # Grayscale
        padded = np.pad(img, ((0, pad_bottom), (0, pad_right)), mode='constant', constant_values=pad_value)
    else:
        # Color image
        padded = np.pad(img, ((0, pad_bottom), (0, pad_right), (0,0)), mode='constant', constant_values=pad_value)
    return padded

def create_gif_from_folder(folder_path, duration=0.5):
    files = [f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')]
    files = [f for f in files if f.split('.')[0].isdigit()]
    files.sort(key=lambda x: int(x.split('.')[0]))

    images = []
    max_height = 0
    max_width = 0

    # First pass: find max dimensions
    for filename in files:
        filepath = os.path.join(folder_path, filename)
        img = imageio.v2.imread(filepath)  # using v2 per warning
        h, w = img.shape[:2]
        if h > max_height:
            max_height = h
        if w > max_width:
            max_width = w
        images.append(img)

    # Pad images to max size
    images_padded = []
    for img in images:
        padded = pad_image_to_size(img, max_height, max_width, pad_value=255)
        images_padded.append(padded)

    folder_name = os.path.basename(os.path.normpath(folder_path))
    output_gif_path = os.path.join(folder_path, f"{folder_name}.gif")

    imageio.mimsave(output_gif_path, images_padded, duration=duration)
    print(f"Saved GIF: {output_gif_path}")

# Example usage:
create_gif_from_folder(r'F:\Documents\testfile\edges\outputpanorama', duration=0.5)
