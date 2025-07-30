import socket
import datetime
import pygetwindow as gw
import time
import pyautogui

pyautogui.PAUSE = 0.1

SCANNING = 1
NON_SCANNING = 0

CURRENT_FOCAL_VALUE = 50
SCANNING_STATUS = False
CURRENT_TAB = "main"
busy = False

def focus_lumedica_window():
    window_title = "Lumedica OQ LabScope"
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

def parse_message(message):
    parts = [part.strip() for part in message.strip().split(',') if part.strip()]
    print(f"Parsed message parts: {parts}")
    
    if not parts:
        raise ValueError("Empty command received")

    function_name = parts[0]
    args = parts[1:]  # Everything else is considered as arguments
    return function_name, args


def start_scan(args):
    time.sleep(0.1)
    # Start scan
    pyautogui.moveTo(157, 259)
    pyautogui.click()
    
    #initialization wait
    time.sleep(1.5)
    SCANNING_STATUS = True

        
def stop_scan( args):
    time.sleep(0.1)
    # Stop Scan
    pyautogui.moveTo(157, 259)
    pyautogui.click()
    SCANNING_STATUS = False
    
def review( args ):
    # Review
    time.sleep(0.1)
    pyautogui.moveTo(191, 177)
    pyautogui.click()
    current_tab = "review"

def return_to_main(args):   
    # Return to main
    time.sleep(0.2)
    pyautogui.moveTo(67, 178)
    pyautogui.click()
    #
    current_tab = "main"

def enter_folder_name(folder_name):
    time.sleep(0.1)
    print(f"entering folder name {folder_name}")
    # Write Folder
    pyautogui.moveTo(1363, 96)
    pyautogui.click()
    time.sleep(0.1)
    pyautogui.hotkey('ctrl', 'a')
    pyautogui.press('backspace')
    pyautogui.write(folder_name, interval =0.1)
    pyautogui.press('enter')
    
    

def enter_filename(final_filename):
    time.sleep(0.1)
    print(f"entering filename {final_filename}")
    # Write filename
    pyautogui.moveTo(1672, 97)
    pyautogui.click()
    pyautogui.hotkey('ctrl', 'a')
    pyautogui.press('backspace')
    pyautogui.write(str(final_filename), interval =0.1)
    pyautogui.press('enter')


def save_single_image(args):
    time.sleep(0.1)
    # Save all images
    pyautogui.moveTo(156, 337)
    pyautogui.click()

    # Esperar wait_save segundos
    time.sleep(1)
    
def save_all_images(args):
    time.sleep(0.1)
    # Save all images
    pyautogui.moveTo(158, 500)
    pyautogui.click()

    # Esperar wait_save segundos
    time.sleep(2)

def change_focal_value(args):
    print(f"change_focal_value {args=}")
    x = 0.0
    try:
        x = float(args[0])
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
    
    return_to_main(args)
    time.sleep(1)
    
    # Modify focal value
    print("moving to", y_coordinate)
    pyautogui.moveTo(345, y_coordinate)
    pyautogui.click()

    # Esperar wait_save segundos
    time.sleep(1)
    
    
    
def simple_scan(args):
    print(f"simple_scan {args=}")
    folder_name = args[0]
    final_filename = args[1]
    wait_save = 1
    print("starting simple scan...")
    
    
    #enter folder name
    enter_folder_name(folder_name)
    
    #enter filename 
    enter_filename(final_filename)
    
    # Start scan
    start_scan(args)
    
    # Stop Scan
    stop_scan(args)
    
    # Review
    review(args)
    
    
    # Save all images
    save_single_image(args)

    # Return to main
    return_to_main(args)
            
def scan_3d(folder_name, final_filename , wait_save = 15, wait_capture = 65):
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
        pyautogui.write(str(final_filename))
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
        
def get_scan_function(function_name):
    function_map = {
        "scan_3d": scan_3d,
        "simple_scan": simple_scan,
        "start_scan": start_scan,
        "stop_scan": stop_scan,
        "enter_folder_name": enter_folder_name,
        "enter_filename": enter_filename,
        "review": review,
        "return_to_main": return_to_main,
        "save_single_image": save_single_image,
        "save_all_images": save_all_images,
        "change_focal_value": change_focal_value,
        "continue": "continue",
    }
    return function_map.get(function_name)

    

def run_function_with_args(scan_func, args):
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
    print(f"Running {scan_func.__name__} with arguments: {args}")
    #scan_func(typed_args)
    scan_func(args)

def separate_multiple_functions(message):
    if not "\n" in message:
        m_list = [message]
        return m_list
    parts = [part.strip() for part in message.strip().split('\n') if part.strip()]
    print(f"Messages: {parts}")
    return parts
    
def get_status():
    status_dict = {"scanning": SCANNING_STATUS,
                   "focal_value": CURRENT_FOCAL_VALUE,
                   "current_tab": CURRENT_TAB
                  }
    message = str(str(status_dict).encode('utf-8'))
    return message
    
def start_server():
    global busy 
    
    focus_lumedica_window()
    time.sleep(2)  # Let it focus

    host = '0.0.0.0'
    port = 12345
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(5)
    print(f"Server listening on port {port}...")
    
    while True:
        try:
        
            client_socket, client_address = server_socket.accept()
            print(f"Connection from {client_address}")
            while True:
                
                initial_message = client_socket.recv(1024).decode('utf-8')
                messages = separate_multiple_functions(initial_message)
                
                for message in messages:
                    print(f"Message received: {message}")

                    try:
                        messages = separate_multiple_functions(message)
                        function_name, args = parse_message(message)
                        scan_func = get_scan_function(function_name)
                        if not scan_func:
                            print(f"Unknown function: {function_name}")
                        if scan_func == "continue":
                            status = get_status().encode("utf-8")
                            client_socket.sendall(status)
                            busy = False
                            continue
                            
                        busy = True
                        run_function_with_args(scan_func, args)
                        client_socket.sendall(b"ack")
                            
                        
                    except Exception as e:
                        error_msg = f"ERROR: {e}"
                        print(error_msg)
                        
                        client_socket.sendall(error_msg.encode())

            client_socket.close()


        except KeyboardInterrupt:
            print("\nShutting down server...")
        except Exception as e:
            print(f"{e}\nShutting down server...")
        
    server_socket.close()


if __name__ == "__main__":
    start_server()
    #just for testing: simple_scan(folder_name="testf", final_filename="final", wait_save = 8)