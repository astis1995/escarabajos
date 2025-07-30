import pyautogui
import time
import os

def clear_console():
    # Cross-platform console clear
    os.system('cls' if os.name == 'nt' else 'clear')

# Get the screen width and height
screen_width, screen_height = pyautogui.size()

try:
    while True:
        # Get current mouse position
        x, y = pyautogui.position()
        
        # Calculate percentage
        x_percent = (x / screen_width) * 100
        y_percent = (y / screen_height) * 100

        clear_console()
        print(f"Cursor Position: ({x}, {y})")
        print(f"Screen Size: {screen_width} x {screen_height}")
        print(f"Percentage Position: ({x_percent:.2f}%, {y_percent:.2f}%)")

        time.sleep(0.1)  # update every 100ms

except KeyboardInterrupt:
    print("\nStopped by user.")
