import pyautogui
import time
import os

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

# Get screen size
screen_width, screen_height = pyautogui.size()

try:
    while True:
        # Get current mouse position
        x, y = pyautogui.position()

        # Calculate position as screen percentage
        x_percent = (x / screen_width) * 100
        y_percent = (y / screen_height) * 100

        # Clear and print updated info
        clear_console()
        print(f"Mouse Position:")
        print(f"  - Pixels: x = {x}, y = {y}")
        print(f"  - Percent: x = {x_percent:.2f}%, y = {y_percent:.2f}%")

        time.sleep(0.1)

except KeyboardInterrupt:
    print("\nStopped.")
