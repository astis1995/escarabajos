#avantes autoclicker
import socket
import datetime
import pygetwindow as gw
import time
import pyautogui
from pywinauto.application import Application
from pywinauto import Desktop
from window_ocr import search_window
from enum import Enum, auto
import pyperclip
from pathlib import Path
import re
from server_abstract import ServerAbstract
pyautogui.PAUSE = 0.1

SCANNING = 1
NON_SCANNING = 0

ROOT_FOLDER = r"C:\Users\Labo402\Avantes\AvaSoft8"

# === Avantes State ===

class SpectrometerState(Enum):
    IDLE = auto()
    SCANNING= auto()


spectrometer_state = SpectrometerState.IDLE

class ReferenceState(Enum):
    NO_REFERENCE = auto()
    REFERENCE_OK= auto()

reference_state = ReferenceState.NO_REFERENCE

class DarkState(Enum):
    NO_DARK = auto()
    DARK_OK= auto()

dark_state = DarkState.NO_DARK

class ScanMode(Enum):
    SCOPE = auto()
    TRANSMITTANCE = auto()
    REFLECTANCE = auto()
    ABSORPTANCE = auto()

scan_mode = ScanMode.SCOPE

#Server Class

class Spectrometer_Server(ServerAbstract):
    
    def process_instruction(self, instruction_id, instruction):
        print(f"{instruction=}")
        function_name, args = parse_message(instruction)
        scan_func = get_scan_function(function_name)
        run_function_with_args(scan_func, args)
        status = self.get_status()
        return status
    
    def get_status(self):
        return get_spectrometer_states()
        
    
def get_spectrometer_states():
    states = f"{spectrometer_state},{dark_state},{reference_state},{scan_mode}"
    return states

def focus_avantes_window():
    time.sleep(1)
    window_title = "AvaSoft 8"
    try:
        app = Application().connect(title= window_title)
        app.top_window().set_focus()
    except Exception as e:
        print(e)
        
        
def focus_avantes_window2():
    time.sleep(1)
    window_title = "AvaSoft 8"
    try:
        win = gw.getWindowsWithTitle(window_title)
        if win:
            lumedica_window = win[0]
            print("minimizing")
            lumedica_window.minimize()  # Ensure minimized first
            time.sleep(0.5)
            print("restoring")
            lumedica_window.restore()   # Restore to bring it to foreground
            time.sleep(0.5)
            print("activating")
            lumedica_window.activate()  # Focus it
            print(f"Focused window: {window_title}")
        else:
            print(f"Window '{window_title}' not found.")
    except Exception as e:
        print(f"Error focusing window: {e}")


def get_today_str():
    return datetime.datetime.now().strftime("%Y-%m-%d")

def parse_message(message):
    parts = [part.strip() for part in message.strip().split(',') if part.strip()]
    print(f"Parsed message parts: {parts}")
    
    if not parts:
        raise ValueError("Empty command received")

    function_name = parts[0]
    args = parts[1:]  # Everything else is considered as arguments
    return function_name, args

def go_home():
    time.sleep(0.5)
    # go to menu home
    pyautogui.moveTo(89, 38)
    pyautogui.click()
    
def start_scan(args):
    go_home()
    
    time.sleep(0.1)
    # Start scan
    pyautogui.moveTo(26, 72)
    pyautogui.click()
    
    #update status
    spectrometer_state = SpectrometerState.SCANNING

        
def stop_scan( args):
    go_home()
    time.sleep(0.1)
    # Stop Scan
    pyautogui.moveTo(26, 72)
    pyautogui.click()
    
    #update status
    spectrometer_state = SpectrometerState.IDLE

def simple_scan( args):
    """Scans as is and saves the data"""
    focus_avantes_window()
    if spectrometer_state != SpectrometerState.SCANNING:
        start_scan(args)
    #create_new_folder(args)
    stop_scan(args)
    save_spectra(args)
    
    
def set_dark(args):   
    # Return to main
    time.sleep(0.1)
    pyautogui.moveTo(155, 70)
    pyautogui.click()
    
    #update status
    dark_state = DarkState.DARK_OK

def set_reference(args):   
    # Return to main
    time.sleep(0.1)
    pyautogui.moveTo(213, 72)
    pyautogui.click()
    
    #update status
    dark_state = DarkState.REFERENCE_OK
    
def transmittance_mode(args):   
    # Return to main
    time.sleep(0.1)
    pyautogui.moveTo(561, 205)
    pyautogui.click()
    
    #update status
    scan_mode = ScanMode.TRANSMITTANCE

def reflectance_mode(args):   
    # Return to main
    time.sleep(0.1)
    pyautogui.moveTo(466, 236)
    pyautogui.click()
    
    #update status
    scan_mode = ScanMode.REFLECTANCE

def absorptance_mode(args):   
    # Return to main
    time.sleep(0.1)
    pyautogui.moveTo(531, 209)
    pyautogui.click()
    
    #update status
    scan_mode = ScanMode.ABSORPTANCE

def scope_mode(args):   
    # Return to main
    time.sleep(0.1)
    pyautogui.moveTo(531, 209)
    pyautogui.click()
    
    #update status
    scan_mode = ScanMode.SCOPE


def enter_folder_name(args):
    time.sleep(0.1)
    folder_name = args[0]
    print(f"entering folder name {folder_name}")
    # Write Folder
    pyautogui.moveTo(1363, 96)
    pyautogui.click()
    time.sleep(0.1)
    pyautogui.hotkey('ctrl', 'a')
    pyautogui.press('backspace')
    pyautogui.write(folder_name, interval =0.1)
    pyautogui.press('enter')
    
    

def enter_filename(args):
    time.sleep(0.1)
    final_filename = args[0]
    print(f"entering filename {final_filename}")
    # Write filename
    pyautogui.moveTo(1672, 97)
    pyautogui.click()
    pyautogui.hotkey('ctrl', 'a')
    pyautogui.press('backspace')
    pyautogui.write(str(final_filename), interval =0.1)
    pyautogui.press('enter')

def go_to_folder(folder):
    #to be used when saving, right after clicking enter to skip comments
    root = r"C:\Users\Labo402\Avantes\AvaSoft8"
    #click address bar
    pyautogui.moveTo(796,238)
    pyautogui.click()
    pyautogui.hotkey('ctrl', 'a')
    pyautogui.press('backspace')
    pyautogui.write(str(folder), interval =0.1)
    pyautogui.press('enter')
    
def create_new_folder(args):
    print(args)
    final_filename = args[5]  # 6th element
    folder_name = args[4]     # 5th element

    time.sleep(0.1)  # Initial buffer

    # Simulated "save" click
    pyautogui.moveTo(342, 206)
    pyautogui.click()
    time.sleep(0.3)
    
    #skip comments
    pyautogui.press('enter')
    time.sleep(0.1)
    
    # Copy what is in the address bar
    pyautogui.moveTo(803, 238)
    pyautogui.click()
    time.sleep(0.2)

    pyautogui.hotkey('ctrl', 'c')
    time.sleep(0.3)  # Give clipboard time to update

    clipboard_content = pyperclip.paste()
    print("Clipboard contains:", clipboard_content)

    # Check if current folder name matches target
    if Path(clipboard_content).name == folder_name:
        return  # No action needed

    # Otherwise, create a new folder
    #go to root
    root = r"C:\Users\Labo402\Avantes\AvaSoft8"
    go_to_folder(root)
    
    #crete new folder
    pyautogui.moveTo(492, 275)
    pyautogui.click()
    time.sleep(0.3)
    
    #delete contents
    pyautogui.press('backspace')
    time.sleep(0.2)
    
    #write folder name
    pyautogui.write(str(folder_name), interval=0.1)
    time.sleep(0.3)
    pyautogui.press('enter')
    
    #close window
    pyautogui.moveTo(1223, 695)
    pyautogui.click()
    time.sleep(0.2)

def do_nothing(args):
    return

def check_if_popup():
        
    # Scan all visible windows
    windows = Desktop(backend="uia").windows()

    # Look for a window with a certain title
    for w in windows:
        if "reemplazarlo" in w.window_text():  # Or partial text from your popup
            return True
            break
    else:
        return False

def save_spectra_ascii(args):
    print(args)
    final_filename = args[1] #5th argument is final_filename
    folder_name = args[0]     # 4th element
    
    #create folder if it does not exist
    folder = Path(ROOT_FOLDER)/folder_name
    folder.mkdir(parents=True, exist_ok=True)
    time.sleep(0.1)
    
    full_name = folder/final_filename
    
    # Start Scan
    start_coor = search_window("Avasoft 8", "Start")
    if start_coor:
        pyautogui.moveTo(start_coor)
        pyautogui.click()
        time.sleep(0.2)
    
    pyautogui.moveTo(search_window("Avasoft 8", "File", xmin = 0.10, xmax= 0.50))
    pyautogui.click()
    time.sleep(0.2)
    
    pyautogui.moveTo(search_window("Avasoft 8", "Export"))
    pyautogui.click()
    time.sleep(0.2)
    
    pyautogui.moveTo(search_window("Avasoft 8", "ASCII"))
    pyautogui.click()
    time.sleep(0.2)
    
    #press enter: OK
    time.sleep(0.5)
    pyautogui.press('enter')
    time.sleep(1)
    
    # Write filename
    #x, y = search_window("Avasoft 8", "Nombre de archivo")
    #x += 100
    #pyautogui.moveTo(x, y)
    #pyautogui.click()
    pyautogui.hotkey('ctrl', 'a')
    pyautogui.press('backspace')
    pyautogui.write(str(full_name), interval =0.1)
    pyautogui.press('enter')
    
    # Esperar wait_save segundos
    time.sleep(1)
    
    # Start Scan
    stop_coor = search_window("Avasoft 8", "Stop")
    if stop_coor:
        pyautogui.moveTo(stop_coor)
        pyautogui.click()
        time.sleep(0.2)
    


def save_spectra(args):
    print(args)
    focus_avantes_window()
    
    final_filename = args[1] #5th argument is final_filename
    folder_name = args[0]     # 5th element
    #create folder if it does not exist
    folder = Path(ROOT_FOLDER)/folder_name
    folder.mkdir(parents=True, exist_ok=True)
    time.sleep(0.1)
    
    # Save all images
    pyautogui.moveTo(338,208)
    pyautogui.click()
    
    #press enter
    time.sleep(0.5)
    pyautogui.press('enter')
    time.sleep(1)
    
    #go to folder 
    go_to_folder(folder)
    
    # Write filename
    pyautogui.moveTo(792, 625)
    pyautogui.click()
    pyautogui.hotkey('ctrl', 'a')
    pyautogui.press('backspace')
    pyautogui.write(str(final_filename), interval =0.1)
    pyautogui.press('enter')
    time.sleep(0.5)
    
    #check if popup (Confirmar guardar como)
    if check_if_popup():
        pyautogui.press('tab') #yes
        pyautogui.press('enter')
        return
    # Click save
    pyautogui.moveTo(1127,693)
    pyautogui.click()
    # Esperar wait_save segundos
    time.sleep(1)
    
    
    
def focus_window(args):
    time.sleep(0.1)
    focus_avantes_window()
    
def get_scan_function(function_name):
    function_map = {
        "focus_window": focus_window,
        "start_scan": start_scan,
        "stop_scan": stop_scan,
        "set_dark": set_dark,
        "set_reference": set_reference,
        "transmittance_mode": transmittance_mode,
        "reflectance_mode": reflectance_mode,
        "absorptance_mode": absorptance_mode,
        "simple_scan": simple_scan,
        "scope_mode": scope_mode,
        "create_new_folder":create_new_folder,
        "save_spectra": save_spectra,
        "do_nothing": do_nothing,
        "save_spectra_ascii":save_spectra_ascii,
    }
    return function_map.get(function_name)

    

def run_function_with_args(scan_func, args):
    # Try converting args to appropriate types
    typed_args = []
    for arg in args:
        if arg is None:
            typed_args.append(arg)
        elif arg.isdigit():
            typed_args.append(int(arg))
        else:
            try:
                typed_args.append(float(arg))
            except ValueError:
                typed_args.append(arg)
    print(f"Running {scan_func.__name__} with arguments: {typed_args}")
    scan_func(typed_args)



def start_server():
        
        
        #focus lumedica window
        focus_avantes_window()
        time.sleep(2)  # Let it focus
        
        #initialize socket
        host = 'localhost'
        port = 11111
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.settimeout(120)
        server_socket.bind((host, port))
        server_socket.listen(5)
        print(f"Server listening on port {port} and host {host}...")
        
        #initialize server 
                
        server = Spectrometer_Server(server_socket)
        
        #flag
        running = True
        
        time.sleep(5)
        try:
            while running:
                try:
                    #accept a connection
                    server.accept()
                    print(f"Connection from {server.client_address}")

                    #start server 
                    server.start()
                except socket.timeout:
                    continue
        except KeyboardInterrupt:
            print("\nShutting down server...")
            running = False
             
        finally:
            server_socket.close()
            print("Socket closed successfully")
            
            
if __name__ == "__main__":
    start_server()
    #just for testing: simple_scan(folder_name="testf", final_filename="final", wait_save = 8)
    #test: save_spectra_ascii(args= ["a", "b","c","d","e","f"])