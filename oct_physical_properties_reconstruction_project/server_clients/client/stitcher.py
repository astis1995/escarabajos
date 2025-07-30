import cv2
import os
import re
from collections import defaultdict

def stitch_images(images):
    # Use OpenCV's built-in Stitcher to align the images
    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    status, stitched_image = stitcher.stitch(images)
    if status == cv2.Stitcher_OK:
        return stitched_image
    else:
        print("Error during stitching:", status)
        return None

def create_panorama_for_each_v(images_folder):
    # Dictionary to store images based on vertical position (v)
    images_by_v = defaultdict(list)

    # Loop through the folder and sort images based on the format 'text_h-v'
    for filename in os.listdir(images_folder):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            # Regex to extract the horizontal (h) and vertical (v) positions from the filename
            match = re.match(r'text_(\d+)-(\d+)', filename)
            if match:
                h = int(match.group(1))  # Horizontal position
                v = int(match.group(2))  # Vertical position
                
                # Load the image using OpenCV
                img_path = os.path.join(images_folder, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load in grayscale (black & white)
                
                # Append the image to the corresponding vertical position (v)
                images_by_v[v].append((h, img))  # Store (h, image)

    # Process each vertical position (v)
    for v, images in images_by_v.items():
        # Sort images for the current v by the horizontal (h) position
        images.sort(key=lambda x: x[0])  # Sort by horizontal value (h)
        
        # Extract just the images in the correct order
        sorted_images = [img for _, img in images]
        
        # Stitch the images together to form the panoramic image
        stitched_image = stitch_images(sorted_images)
        
        if stitched_image is not None:
            # Save the panorama for this v
            cv2.imwrite(f'panorama_v{v}.png', stitched_image)
            print(f'Panorama for v={v} saved as panorama_v{v}.png')

# Example usage:
images_folder = 'path_to_your_image_folder'  # Replace with the folder containing your images
create_panorama_for_each_v(images_folder)
