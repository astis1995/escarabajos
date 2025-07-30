import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def show_luminance_histogram(image_path):
    # Step 1: Load and convert image to grayscale
    image = Image.open(image_path).convert("L")  # "L" = luminance (grayscale)
    grayscale_array = np.array(image)

    # Step 2: Flatten the array to 1D for histogram
    pixel_values = grayscale_array.flatten()

    # Step 3: Plot the histogram
    plt.figure(figsize=(8, 5))
    plt.hist(pixel_values, bins=256, range=(0, 255), color='gray', alpha=0.7, edgecolor='black')
    plt.title("Histogram of Luminance")
    plt.xlabel("Luminance Value (0=Black, 255=White)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    image_path = input("Enter the path to the image: ").strip()
    image_path = r"E:\OCT\coptima-2025-07-03\20250708-153943\stitched\01_stitched.jpg"
    if os.path.exists(image_path):
        
        show_luminance_histogram(image_path)
    else:
        print("❌ File not found.")
