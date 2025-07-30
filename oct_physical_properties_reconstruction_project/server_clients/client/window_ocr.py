import pygetwindow as gw
import pyautogui
import pytesseract
import cv2
import numpy as np
from PIL import Image
import time
import os

# Optional: Set Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Labo402\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

def focus_window(window_title):
    windows = gw.getWindowsWithTitle(window_title)
    if not windows:
        raise Exception(f"Window with title '{window_title}' not found.")
    win = windows[0]
    win.activate()
    time.sleep(1)
    return win

def take_screenshot(region=None):
    screenshot = pyautogui.screenshot(region=region)
    return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

def preprocess_for_ocr(image, upscale_factor=2):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (0, 0), fx=upscale_factor, fy=upscale_factor, interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(resized, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def find_word(image_np, word):
    processed = preprocess_for_ocr(image_np)
    data = pytesseract.image_to_data(processed, output_type=pytesseract.Output.DICT)

    for i in range(len(data['text'])):
        if word.lower() in data['text'][i].lower():
            x = int(data['left'][i] / 2)
            y = int(data['top'][i] / 2)
            w = int(data['width'][i] / 2)
            h = int(data['height'][i] / 2)
            return (x, y, w, h)
    return None

def find_icon(image_np, template_path, threshold=0.8):
    template = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
    if template is None:
        raise Exception(f"Template image not found: {template_path}")
    
    result = cv2.matchTemplate(image_np, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val >= threshold:
        return (max_loc[0], max_loc[1], template.shape[1], template.shape[0])
    return None

def search_window(window_title, search_input, xmin=None, xmax=None, ymin=None, ymax=None):
    win = focus_window(window_title)
    bbox = (win.left, win.top, win.width, win.height)
    screenshot = take_screenshot(region=bbox)

    # Calculate subregion based on percentage bounds
    img_h, img_w = screenshot.shape[:2]
    x_start = int(xmin * img_w) if xmin is not None else 0
    x_end = int(xmax * img_w) if xmax is not None else img_w
    y_start = int(ymin * img_h) if ymin is not None else 0
    y_end = int(ymax * img_h) if ymax is not None else img_h

    # Crop to subregion
    sub_image = screenshot[y_start:y_end, x_start:x_end]

    if os.path.isfile(search_input):
        result = find_icon(sub_image, search_input)
    else:
        result = find_word(sub_image, search_input)

    if result:
        x, y, w, h = result
        absolute_x = bbox[0] + x_start + x
        absolute_y = bbox[1] + y_start + y
        center_x = absolute_x + w // 2
        center_y = absolute_y + h // 2
        print(f"{result=} {center_x=}{center_y=}")
        return center_x, center_y, w, h
    return None


# Run only if executed directly
if __name__ == "__main__":
    window_title = "Avasoft 8"
    search_input = r"C:\Users\Labo402\autoclicker_img"  # Can be image path or string
    result = search_window(window_title, search_input)
    if result:
        center_x, center_y, w, h = result
        print(f"Found at center: ({center_x}, {center_y}) with size ({w}, {h})")
    else:
        print("Not found.")
