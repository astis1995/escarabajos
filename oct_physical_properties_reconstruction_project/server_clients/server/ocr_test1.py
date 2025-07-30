from PIL import ImageGrab, ImageDraw
import pytesseract

# Path to tesseract (Windows users may need to update this)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Step 1: Take screenshot
screenshot = ImageGrab.grab()
screenshot_rgb = screenshot.convert("RGB")

# Step 2: Extract OCR data with bounding boxes
data = pytesseract.image_to_data(screenshot_rgb, output_type=pytesseract.Output.DICT)

# Step 3: Draw rectangles around detected text
draw = ImageDraw.Draw(screenshot_rgb)

print("Detected text and positions:")
for i in range(len(data['text'])):
    text = data['text'][i].strip()
    if text:
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        print(f"Text: '{text}' at (x={x}, y={y}, w={w}, h={h})")
        draw.rectangle([(x, y), (x + w, y + h)], outline="red", width=2)

# Step 4: Save image with rectangles
output_path = "output_with_boxes.png"
screenshot_rgb.save(output_path)
print(f"\nAnnotated screenshot saved as: {output_path}")
