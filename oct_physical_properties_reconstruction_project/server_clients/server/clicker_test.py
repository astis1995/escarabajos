import pyautogui
import time

def simple_scan(folder_name, file_counter, wait_save = 5):
    # Start scan
        pyautogui.moveTo(157, 259)
        pyautogui.click()

        # Esperar wait_capture segundos
        time.sleep(3)

        # Stop Scan
        pyautogui.moveTo(157, 259)
        pyautogui.click()
        
        # Review
        pyautogui.moveTo(191, 177)
        pyautogui.click()
        
        # Write Folder
        pyautogui.moveTo(1363, 96)
        pyautogui.click()
        pyautogui.hotkey('ctrl', 'a')
        pyautogui.press('backspace')
        pyautogui.write(folder_name)
        pyautogui.press('enter')
        
        # Esperar 1 segundos
        time.sleep(1)
        
        # Write filename
        pyautogui.moveTo(1672, 97)
        pyautogui.click()
        pyautogui.hotkey('ctrl', 'a')
        pyautogui.press('backspace')
        pyautogui.write(str(file_counter))
        pyautogui.press('enter')
        
        # Esperar 1 segundos
        time.sleep(1)
        
        # Save all images
        pyautogui.moveTo(158, 500)
        pyautogui.click()

        # Esperar wait_save segundos
        time.sleep(wait_save)

        # Return to main
        pyautogui.moveTo(67, 178)
        pyautogui.click()
            
def scan_3d(folder_name, file_counter , wait_save = 20):
    # Capture Volumen Scan
        pyautogui.moveTo(159, 498)
        pyautogui.click()

        # Esperar wait_capture segundos
        time.sleep(wait_capture)

        # Review
        pyautogui.moveTo(188, 178)
        pyautogui.click()

        # Write Folder
        pyautogui.moveTo(1363, 96)
        pyautogui.click()
        pyautogui.hotkey('ctrl', 'a')
        pyautogui.press('backspace')
        pyautogui.write(folder_name)
        pyautogui.press('enter')
        
        # Esperar 1 segundos
        time.sleep(1)
        
        # Write filename
        pyautogui.moveTo(1672, 97)
        pyautogui.click()
        pyautogui.hotkey('ctrl', 'a')
        pyautogui.press('backspace')
        pyautogui.write(str(file_counter))
        pyautogui.press('enter')
        
        # Esperar 1 segundos
        time.sleep(1)
        
        # Save all images
        pyautogui.moveTo(158, 500)
        pyautogui.click()

        # Esperar wait_save segundos
        time.sleep(wait_save)

        # Return to main
        pyautogui.moveTo(67, 178)
        pyautogui.click()

def automate(folder_name, file_counter_start, wait_capture=65, wait_save=30, repetitions=1):
    file_counter = file_counter_start
    scan_type = simple_scan
    
    for i in range(repetitions):
        print(f"--- Iteración {i+1} ---")
        scan_type(folder_name, file_counter)
        

        file_counter += 1  # incrementar contador de archivo

if __name__ == "__main__":
    folder = input("Nombre del folder: ")
    start_counter = int(input("Número inicial de archivo: "))
    wait_capture_time = int(input("Esperar segundos para captura (default 65): ") or 65)
    wait_save_time = int(input("Esperar segundos para guardar (default 20): ") or 20)
    reps = int(input("Número de repeticiones: "))

    print("Tienes 5 segundos para poner la ventana activa...")
    time.sleep(5)

    automate(folder, start_counter, wait_capture_time, wait_save_time, reps)
