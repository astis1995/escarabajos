import socket
import datetime
import pygetwindow as gw
import time
import pyautogui
from server_abstract import ServerAbstract
import re
from enum import Enum, auto
import traceback
import json
from window_ocr import search_window, search_icon
from gui_file_server import *
#enum classes
import tkinter as tk

file_server = None
import threading

class AlignmentState(Enum):
    HORIZONTAL = auto()
    VERTICAL = auto()
    CROSS = auto()
    CIRCLE = auto()
    
class ScanningState(Enum):
    SCANNING = auto()
    NOT_SCANNING = auto()
    
class TabState(Enum):
    MAIN = auto()
    REVIEW = auto()
    CONFIGURATION = auto()
    ADVANCED = auto()
    
#constants
pyautogui.PAUSE = 0.1

SCANNING = 1
NON_SCANNING = 0

focal_value = 50.0
focus_value = 50.0
power_value = 50.0
main_dynamic_range_bottom = 0.0
review_dynamic_range_bottom = 0.0
scanning_state = ScanningState.NOT_SCANNING
current_tab = TabState.MAIN
alignment = AlignmentState.HORIZONTAL
busy = False
dispersion_compensation_B = 0.0
dispersion_compensation_C = 0.0
current_folder = ""
volume_scan_width = 0.5

#TIME CONSTANTS
MIN_WAIT = 0.1 #seconds

#default values
DEFAULT_FOCAL_VALUE = 50.0
DEFAULT_FOCUS_VALUE = 50.0
DEFAULT_POWER_VALUE = 100.0
DEFAULT_DISPERSION_COMPENSATION_B = 0.00
DEFAULT_DISPERSION_COMPENSATION_C = 5.00
DEFAULT_DINAMIC_RANGE_BOTTOM = 0.0
DEFAULT_ALIGMENT = AlignmentState.HORIZONTAL
LUMEDICA_WINDOW_TITLE = "Lumedica OQ LabScope"
ROOT_DIR = r"C:\Users\Public\Documents\Lumedica\OctEngine\Data"


def setup():
    change_focal_value([DEFAULT_FOCAL_VALUE])
    change_focus_value([DEFAULT_FOCUS_VALUE])
    change_power_value([DEFAULT_POWER_VALUE])
    change_dispersion_compensation_B_value(DEFAULT_DISPERSION_COMPENSATION_B)
    change_dispersion_compensation_C_value(DEFAULT_DISPERSION_COMPENSATION_C)
    set_alignment_scan_type([DEFAULT_ALIGMENT])
    
def focus_lumedica_window():
        window_title = LUMEDICA_WINDOW_TITLE
        try:
            win = gw.getWindowsWithTitle(window_title)
            if win:
                lumedica_window = win[0]
                lumedica_window.minimize()  # Ensure minimized first
                time.sleep(0.2)
                lumedica_window.restore()   # Restore to bring it to foreground
                lumedica_window.activate()  # Focus it
                print(f"Focused window: {window_title}")
            else:
                print(f"Window '{window_title}' not found.")
        except Exception as e:
            print(f"Error focusing window: {e}")
            
def get_today_str():
        return datetime.datetime.now().strftime("%Y-%m-%d")

def check_scanning(running_scan_check = True,  xmin = 0.0, xmax = 0.25, ymin = 0.20, ymax = 0.60):
    
    scanning = (scanning_state == ScanningState.SCANNING) == running_scan_check #modificar
    print("scanning:", running_scan_check, scanning)
    return scanning 
    #####################################################
    main_tab()
    
    #Look for Stop Scan (Scan running)
    #start_scan_coor = search_window(LUMEDICA_WINDOW_TITLE, "Start", xmin, xmax, ymin, ymax)
    #stop_scan_coor = search_window(LUMEDICA_WINDOW_TITLE, "Stop", xmin, xmax, ymin, ymax)
    start_scan_coor = search_icon(LUMEDICA_WINDOW_TITLE, "start_scan", xmin, xmax, ymin, ymax)
    stop_scan_coor = search_icon(LUMEDICA_WINDOW_TITLE, "stop_scan", xmin, xmax, ymin, ymax)
    if start_scan_coor:
        print(f"Start scan found {start_scan_coor}")
    if stop_scan_coor:
        print(f"Stop scan found. ")
    else:
        print("Nothing found")
        
    scan_running = ( not start_scan_coor) and (stop_scan_coor )
    
    #test result
    result = (running_scan_check and scan_running) or (not running_scan_check and not scan_running)
    print(f"Check scanning == {running_scan_check}? and result {result=}.  Current status {scanning_state=}")
    return result

def toggle_scan(args):
        if check_scanning(True):
           stop_scan(args)
           print("stopping scan")
        else:
           start_scan(args)
           print("starting scan")
def start_scan(args):
    #go to main tab
    main_tab()
    
    time.sleep(MIN_WAIT) 
    # Start scan
    # Check if it is already scanning
    
    #if check_scanning(False):
    pyautogui.moveTo(157, 259)
    pyautogui.click()
    
    #initialization wait
    time.sleep(1.5)
    scanning_state = ScanningState.SCANNING

            
def stop_scan( args = None):
    #go to main tab
    main_tab()
        
    time.sleep(MIN_WAIT) 
    
    pyautogui.moveTo(157, 259)
    pyautogui.click()
        
    scanning_state = ScanningState.NOT_SCANNING
    
def review_tab( args ):
    # review_tab
    time.sleep(MIN_WAIT) 
    pyautogui.moveTo(191, 177)
    pyautogui.click()
    current_tab = TabState.REVIEW
    time.sleep(MIN_WAIT) 

def configuration_tab( args ):
    # configuration_tab
    time.sleep(MIN_WAIT) 
    pyautogui.moveTo(306, 179)
    pyautogui.click()
    current_tab = TabState.CONFIGURATION
    time.sleep(1.3)

def advanced_tab( args ):
    # advanced_tab
    time.sleep(MIN_WAIT) 
    pyautogui.moveTo(434, 180)
    pyautogui.click()
    current_tab = TabState.ADVANCED
    time.sleep(0.9)
    
def main_tab(args = None):   
    # Return to main
    time.sleep(0.2)
    pyautogui.moveTo(67, 178)
    pyautogui.click()
    #
    current_tab = TabState.MAIN
    time.sleep(MIN_WAIT) 

def enter_folder_name(args):
    folder_name = args[0]
    time.sleep(MIN_WAIT) 
    print(f"entering folder name {folder_name}")
    #go to main tab
    main_tab()
    
    #check if it is scanning
    if check_scanning():
        stop_scan() #and stop it
        
    # Write Folder
    pyautogui.moveTo(1363, 96)
    pyautogui.click()
    print(f"writing")
    time.sleep(MIN_WAIT) 
    pyautogui.hotkey('ctrl', 'a')
    pyautogui.press('backspace')
    pyautogui.write(folder_name, interval =0.1)
    pyautogui.press('enter')
    
    #set current folder
    current_folder = folder_name
    

def enter_filename(args):
    final_filename = args[0]
    time.sleep(MIN_WAIT) 
    print(f"entering filename {final_filename}")
    # Write filename
    pyautogui.moveTo(1672, 97)
    pyautogui.click()
    pyautogui.hotkey('ctrl', 'a')
    pyautogui.press('backspace')
    pyautogui.write(str(final_filename), interval =0.1)
    pyautogui.press('enter')

def save_configuration_file(folder = None):
    configuration_dict = {
                            "focal_value" : focal_value,
                            "focus_value" : focus_value,
                            "power_value" : power_value,
                            "main_dynamic_range_bottom" : main_dynamic_range_bottom,
                            "review_dynamic_range_bottom" : review_dynamic_range_bottom,
                            "scanning_state" : scanning_state,
                            "current_tab" : current_tab,
                            "alignment" : alignment,
                            "busy" : busy,
                            "folder": ".",
                            "dispersion_compensation_B" : dispersion_compensation_B,
                            "dispersion_compensation_C" : dispersion_compensation_C,
                            }
    if not folder:
        folder = configuration_dict["folder"]
    
    readable_time = time.strftime("%Y-%m-%d %H.%M.%S, time.localtime()")
    if folder:
        config_file_name = f"{folder}\\{readable_time}.config"
    else: 
        config_file_name = f"{readable_time}.config"
    try:
        with open(config_file_name, "w") as f:
            f.write(f"{readable_time}\n")
            for key, value in configuration_dict.items():
                f.write(f"{key}\t{value}\n")
    except Exception as e:
        print(e)

def save_single_image(args):
    
    time.sleep(MIN_WAIT) 
    # Save all images
    pyautogui.moveTo(156, 337)
    pyautogui.click()

    # Esperar wait_save segundos
    time.sleep(0.2)
    
def save_all_images(args):
    time.sleep(MIN_WAIT) 
    # Save all images
    pyautogui.moveTo(158, 500)
    pyautogui.click()

    # Esperar wait_save segundos
    time.sleep(2)

def change_main_dynamic_range_bottom_value(args):
    print(f"change_main_dynamic_range_bottom_value {args=}")
    f = 0.0
    try:
        f = float(args[0])
    except Exception as e:
        print(e)
        return
    x_min, y_min = 1142.0, 960.0
    x_max, y_max = 1142.0, 870.0 
    
    #unchanging coordinate
    
    fixed_coor = x_max 
    #Slider limits
    slider_min, slider_max = 0.0, 1.0
    #x = mf + b , b = x-mf
    m = (y_max - y_min)/(slider_max - slider_min)
    b = y_max - m * slider_max
    
    #power is p
    variable_coordinate = m * f + b 
    
    #go to main tab
    
    main_tab()
    
    # Modify power value
    print("moving to", variable_coordinate)
    pyautogui.moveTo(fixed_coor, variable_coordinate)
    pyautogui.click()

    # Esperar wait_save segundos
    time.sleep(MIN_WAIT) 
    
    #Update 
    main_dynamic_range_bottom = f 
    
def change_review_dynamic_range_bottom_value(args):
    print(f"change_review_dynamic_range_bottom_value {args=}")
    f = 0.0
    try:
        f = float(args[0])
    except Exception as e:
        print(e)
        return
    x_min, y_min = 344.0, 866.0
    x_max, y_max = 344.0, 260.0 
    
    #unchanging coordinate
    
    fixed_coor = x_max 
    #Slider limits
    slider_min, slider_max = 0.0, 1.0
    #x = mf + b , b = x-mf
    m = (y_max - y_min)/(slider_max - slider_min)
    b = y_max - m * slider_max
    
    #power is p
    variable_coordinate = m * f + b 
    
    
    #go to review 
    review_tab(args)
    
    # Modify power value
    print("moving to", variable_coordinate)
    pyautogui.moveTo(fixed_coor, variable_coordinate)
    pyautogui.click()

    # Esperar MIN_WAIT segundos
    time.sleep(MIN_WAIT) 
    
    
    #Update 
    review_dynamic_range_bottom = f 
    
def change_focal_value(args):
    print(f"change_focal_value {args=}")
    x = 0.0
    try:
        x = float(args[0])
        print("focal value parsed", x)
    except Exception as e:
        print(e)
        return
    y_100 = 259.0
    y_0 = 871.0
    x_0 =0.0
    x_100 = 100.0
    m = (y_100 - y_0)/(x_100 -x_0)
    b = y_100 - m * x_100
    
    y_coordinate =m * x + b 
    
    main_tab()
    
    # Modify focal value
    print("moving to", y_coordinate)
    pyautogui.moveTo(345, y_coordinate)
    pyautogui.click()

    # Esperar wait_save segundos
    time.sleep(0.3)
    
    #Update 
    focal_value = x 

def get_slider_coordinate(x_coor_min, y_coor_min, x_coor_max, y_coor_max, min_value, max_value, direction = "horizontal", desired_value = 0.0):
    # coordinate(value) = m* value + b then: m = (coor2-coor1)/(value2 -value1) and b = coordinate2-m*value2
    
    if direction == "horizontal":
        m = (x_coor_max -x_coor_min)/(max_value-min_value)
        b = x_coor_max - m * max_value
        frozen_coordinate = y_coor_min #or max, they should be the same
        desired_coordinate = m * desired_value + b 
        return (desired_coordinate, frozen_coordinate)
    if direction == "vertical":    
        m = (y_coor_max -y_coor_min)/(max_value-min_value)
        b = y_coor_max - m * max_value
        frozen_coordinate = x_coor_min #or max, they should be the same
        desired_coordinate = m * desired_value + b 
        return( frozen_coordinate, desired_coordinate)
    print("Error, direction not supported", direction)
    return None
    
def change_volume_scan_width(args):
    print(f"change_volume_scan_width {args=}")
    width = 0.5
    try:
        width = float(args[0])
    except Exception as e:
        print(e)
        return
    width = args[0]
    
    x_coor_min, y_coor_min, min_value  = 795,563,0.0
    x_coor_max, y_coor_max, max_value = 1225, 563, 1.0
    direction = "horizontal"
    desired_value = width 
    
    coordinates = get_slider_coordinate(x_coor_min, y_coor_min, x_coor_max, y_coor_max, min_value, max_value, direction , desired_value)
    
    #go to configuration_tab
    
    configuration_tab(args)
    
    # Modify value
    pyautogui.moveTo(coordinates)
    pyautogui.click()
    
    # Esperar wait_save segundos
    time.sleep(0.3)
    
    #Return to main
    main_tab()
    
    #Update 
    volume_scan_width = width

def change_power_value(args):
    print(f"change_power_value {args=}")
    p = 0.0
    try:
        p = float(args[0])
    except Exception as e:
        print(e)
        return
    x_0, y_0 = 1415.0, 945.0
    x_100, y_100 = 1836.0, 945.0 
    
    m = (x_100 - x_0)/(100.0 - 0.0)
    b = x_100 - m * 100
    
    #power is p
    #x = mp + b , b = x-mp
    x_coordinate = m * p + b 
    
    #go to advanced tab
    
    advanced_tab(args)
    
    # Modify power value
    print("moving to", x_coordinate)
    pyautogui.moveTo(x_coordinate, 945)
    pyautogui.click()

    # Esperar wait_save segundos
    time.sleep(0.3)
    
    #Return to main
    main_tab()
    
    #Update 
    power_value = p
    
def change_focus_value(args):
    print(f"change_focus_value {args=}")
    f = 0.0
    try:
        f = float(args[0])
    except Exception as e:
        print(e)
        return
    x_0, y_0 = 802.0, 876.0
    x_100, y_100 = 1223.0, 876.0 
    
    #x = mf + b , b = x-mf
    m = (x_100 - x_0)/(100.0 - 0.0)
    b = x_100 - m * 100
    
    #power is p
    x_coordinate = m * f + b 
    
    #go to advanced tab
    
    advanced_tab(args)
    
    # Modify power value
    print("moving to", x_coordinate)
    pyautogui.moveTo(x_coordinate, 876)
    pyautogui.click()

    # Esperar wait_save segundos
    time.sleep(0.3)
    
    #Return to main
    main_tab()
    
    #Update 
    focus_value = f 
    
def change_dispersion_compensation_B_value(args):
    print(f"change_dispersion_compensation_B_value {args=}")
    f = 0.0
    try:
        f = float(args[0])
    except Exception as e:
        print(e)
        return
    x_min, y_min = 436.0, 960.0
    x_max, y_max = 707.0, 960.0 
    
    #unchanging coordinate
    
    fixed_coor = y_max 
    #Slider limits
    slider_min, slider_max = -50.0, 25.0
    #x = mf + b , b = x-mf
    m = (x_max - x_min)/(slider_max - slider_min)
    b = x_max - m * slider_max
    
    #power is p
    x_coordinate = m * f + b 
    
    #go to main tab
    
    main_tab()
    
    # Modify power value
    print("moving to", x_coordinate)
    pyautogui.moveTo(x_coordinate, fixed_coor)
    pyautogui.click()

    # Esperar wait_save segundos
    time.sleep(MIN_WAIT) 
    
    #Update 
    dispersion_compensation_B = f 
    
    
def change_dispersion_compensation_C_value(args):
    print(f"change_dispersion_compensation_C_value {args=}")
    f = 0.0
    try:
        f = float(args[0])
    except Exception as e:
        print(e)
        return
    x_min, y_min = 777.0, 958.0
    x_max, y_max = 1045.0, 958.0 
    
    #unchanging coordinate
    
    fixed_coor = y_max 
    #Slider limits
    slider_min, slider_max = -50.0, 25.0
    #x = mf + b , b = x-mf
    m = (x_max - x_min)/(slider_max - slider_min)
    b = x_max - m * slider_max
    
    #power is p
    x_coordinate = m * f + b 
    
    #go to main tab
    
    main_tab()
    
    # Modify power value
    print("moving to", x_coordinate)
    pyautogui.moveTo(x_coordinate, fixed_coor)
    pyautogui.click()

    # Esperar wait_save segundos
    time.sleep(MIN_WAIT) 
    
    #Update 
    dispersion_compensation_B = c
    
def simple_scan(args):
    print(f"Starting simple_scan {args=}")
    folder_name = args[0]
    final_filename = args[1]
    wait_save = 1
    print("starting simple scan...")
    
    #if current tab is not main
    if current_tab != TabState.MAIN:
        main_tab() #go to main
        
    #if it is scanning, stop current scan
    if check_scanning(True):
       stop_scan(args)
    
    
    #save configuration file
    save_configuration_file()
    
    #enter filename 
    enter_filename([final_filename])
    
    # Start scan
    start_scan(args)
    
    # Stop Scan
    stop_scan()
    
    # review_tab
    review_tab(args)
    
    # Save all images
    save_single_image(args)

    # Return to main
    main_tab()
            
def scan_3d(args):
    filename = args[1]
    folder = args[0]
    wait_save = 15
    wait_capture = 65
    #go to main tab
    main_tab()
    
    #   
    if check_scanning(True):
       stop_scan(args)
       
    # Capture Volumen Scan
    pyautogui.moveTo(159, 498)
    pyautogui.click()

    # Esperar wait_capture segundos
    time.sleep(wait_capture)
    
    # review_tab
    pyautogui.moveTo(188, 178)
    pyautogui.click()

    # Esperar 1 segundos
    time.sleep(0.2)
    
    # Save all images
    pyautogui.moveTo(158, 500)
    pyautogui.click()

    # Esperar wait_save segundos
    time.sleep(wait_save)

    # Return to main
    pyautogui.moveTo(67, 178)
    pyautogui.click()

def continue_function(args):
    print("Continue function: does nothing")
    
def get_function(function_name):
    function_map = {
        "scan_3d": scan_3d,
        "simple_scan": simple_scan,
        "start_scan": start_scan,
        "stop_scan": stop_scan,
        "enter_folder_name": enter_folder_name,
        "enter_filename": enter_filename,
        "review_tab": review_tab,
        "main_tab": main_tab,
        "save_single_image": save_single_image,
        "save_all_images": save_all_images,
        "change_focal_value": change_focal_value,
        "change_focus_value": change_focus_value,
        "change_power_value": change_power_value,
        "continue": continue_function,
        "change_dispersion_compensation_C_value":change_dispersion_compensation_C_value,
        "change_dispersion_compensation_B_value":change_dispersion_compensation_B_value,
        "set_alignment_scan_type":set_alignment_scan_type,
        "change_review_dynamic_range_bottom_value":change_review_dynamic_range_bottom_value,
        "change_main_dynamic_range_bottom_value":change_main_dynamic_range_bottom_value,
        "send_folder_to_client":send_folder_to_client,
        "change_volume_scan_width":change_volume_scan_width, 
    }
    return function_map.get(function_name)

def set_alignment_scan_type(args):
    
    scan_type = str(args[0])
    #go to configuration tab
    configuration_tab(args)
    
    if "scan_type" == "horizontal":
        pyautogui.moveTo(85, 313)
        pyautogui.click()
        alignment = AlignmentState.HORIZONTAL
        time.sleep(MIN_WAIT)
        
    elif "scan_type" == "vertical":
        pyautogui.moveTo(85, 341)
        pyautogui.click()
        alignment = AlignmentState.VERTICAL
        time.sleep(MIN_WAIT)
    elif "scan_type" == "cross":
        pyautogui.moveTo(85, 370)
        pyautogui.click()
        alignment = AlignmentState.CROSS
        time.sleep(MIN_WAIT)
    elif "scan_type" == "circle":
        pyautogui.moveTo(85, 399)
        pyautogui.click()
        alignment = AlignmentState.CIRCLE
        time.sleep(MIN_WAIT)
    else:
        print(f"scan_type invalid: {scan_type}")
        return

     

def run_function_with_args(function, args):
    # Try converting args to appropriate types
    typed_args = []
    
    for arg in args:
        print(f"current arg {arg}")
        if arg is None:
            typed_args.append(arg)
        elif arg.isdigit():
            typed_args.append(int(arg))
        else:
            try:
                typed_args.append(arg)
            except ValueError:
                typed_args.append(arg)
    print(f"Running {function.__name__} with arguments: {args}")
    #function(typed_args)
    function(args)

def separate_multiple_functions(message):
    if not "\n" in message:
        m_list = [message]
        return m_list
    parts = [part.strip() for part in message.strip().split('\n') if part.strip()]
    print(f"Messages: {parts}")
    return parts

ENUM_REGISTRY = {
    "AlignmentState": AlignmentState,
    "ScanningState":ScanningState,
    "TabState": TabState,
}
def encode_data(data):
    def convert(obj):
        
        if isinstance(obj, Enum):
            return {"__enum__": f"{obj.__class__.__name__}.{obj.name}"}
        return obj
    return json.dumps(data, default=convert).encode("utf-8")
    
def get_status_2():
        status_dict = {"scanning": scanning_state,
                       "alignment": focus_value,
                       "focal_value": focal_value,
                       "current_tab": current_tab,
                       "power_value": power_value,
                       "focus_value": focus_value,
                       "dispersion_compensation_B": dispersion_compensation_B,
                       "dispersion_compensation_C": dispersion_compensation_C,
                      }
        #convert to json
        serialized = encode_data(status_dict)
        
        #message = str(status_dict)
        return serialized

def parse_message( message):
        print(f"message received",message)
        parts = [part.strip() for part in message.strip().split(',') if part.strip()]
        print(f"Parsed message parts: {parts}")
        
        if not parts:
            raise ValueError("Empty command received")

        function_name = parts[0]
        args = parts[1:]  # Everything else is considered as arguments
        return function_name, args

def send_folder_to_client(args):
    global file_server, current_folder
    
    if not file_server:
        threading.Thread(target=start_file_server, daemon=True).start()
    folder = f"{ROOT_DIR}\\{current_folder}"
    try:
        arg_folder = args[0]
        print("arg_folder", arg_folder)
        folder = f"{ROOT_DIR}\\{arg_folder}"
    except Exception as e:
        print("Error:", e, " enviando current_folder", current_folder)
        
    print("sending folder...")
    file_server.send_zip_folder_to_client(folder)
    
    
class OCT_Server(ServerAbstract):
    

    def process_instruction(self, instruction_id, instruction):
        
        print(f"Processing {instruction=}")
        function_name, args = parse_message(instruction)
        print(f"Parsed function {function_name=} with args {args=}")
        scan_func = get_function(function_name)
        print(f"Trying to run {function_name=} with args {args=}")
        run_function_with_args(scan_func, args)
        print(f"Getting current status ")
        status = self.get_status()
        print(f"Returning current status {status}")
        return status
    
    def get_status(self):
        return get_status_2()





def start_file_server():
    global file_server
    try:
        print("starting file server...")
        root = tk.Tk()
        file_server = FileServerGUI(root)
        root.mainloop()
    except Exception as e:
        print(e)
        
def start_server():
        global busy, file_server
        
        #start file server
        threading.Thread(target=start_file_server, daemon=True).start()

        
        #focus lumedica window
        focus_lumedica_window()
        time.sleep(2)  # Let it focus
        
        #initialize socket
        host = '192.168.188.2'
        port = 12345
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.settimeout(120)
        server_socket.bind((host, port))
        server_socket.listen(5)
        print(f"Server listening on port {port} and host {host}...")
        
        #initialize server 
                
        server = OCT_Server(server_socket)
        
        #flag
        running = True
        
        
        
        try:
            while running:
                try:
                    #accept a connection
                    server.accept()
                    print(f"Connection from {server.client_address}")
                    #accept file server connection
                    if file_server:
                        file_server.wait_for_connection()
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