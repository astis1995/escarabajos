import pyautogui
import time

# Set the target position (change this to your desired coordinates)
target_x = 500
target_y = 300

# Pause before moving (gives you time to switch to another window if needed)
print("Moving cursor in 3 seconds...")
time.sleep(3)

# Move the cursor smoothly over 0.5 seconds
pyautogui.moveTo(target_x, target_y, duration=0.5)

# Perform a left click
pyautogui.click()

print(f"Clicked at ({target_x}, {target_y})")
