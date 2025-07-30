import tkinter as tk
from threading import Thread, Event
import threading
import serial
import time
import os
import socket
from pynput import keyboard
from enum import Enum, auto
from spectrometer_client import *
from oct_client import *
from oct_client import OCTClientErrorCode
from arduino_stepper_controller import ArduinoStepperController
from cnc_server import get_today_str, real_mm_to_program_units, CNC_Server
import datetime
import time
import pyautogui
import numpy as np
from state_classes import *
from gui_file_client import FileClientGUI

# === CNC Constants ===
PROGRAM_UNITS_PER_REAL_MM_X   = 1.631
PROGRAM_UNITS_PER_REAL_MM_Y   = 1.609
PROGRAM_UNITS_PER_REAL_MM_Z   = 1.593
PROGRAM_UNITS_PER_REAL_MM = (PROGRAM_UNITS_PER_REAL_MM_X+PROGRAM_UNITS_PER_REAL_MM_Y+PROGRAM_UNITS_PER_REAL_MM_Z)/3
rotation_angle = 1.0

CURRENT_POSITION_FILE = "current_position.txt"
HOME_POSITION_FILE = "home_position.txt"
ORIGIN_POSITION_FILE = "absolute_position_origin.txt"

PROGRAM_UNITS_PER_HORIZONTAL_FRAME = 4.322
REAL_MM_PER_HORIZONTAL_FRAME = PROGRAM_UNITS_PER_HORIZONTAL_FRAME/ PROGRAM_UNITS_PER_REAL_MM_X
calibration = True


# === OCT Constants ===
OCT_FOCUS_DISTANCE_MM = 15

# === SPECTROMETER Constants ===
REFLECTANCE_PROBE_FOCUS_DISTANCE_MM_TOP = 9
REFLECTANCE_PROBE_FOCUS_DISTANCE_MM_BOTTOM = 11
REFLECTANCE_PROBE_FOCUS_DISTANCE_PROGRAM_UNITS_TOP = 0
REFLECTANCE_PROBE_FOCUS_DISTANCE_PROGRAM_UNITS_BOTTOM = 7.647 #Debe subir la sonda

REFLECTANCE_PROBE_OCT_X_PROGRAM_DISTANCE = -0.734 #mm programa. positivo: hay que moverse a la izquierda, negativo a la derecha
REFLECTANCE_PROBE_OCT_Y_PROGRAM_DISTANCE = 66.774 #mm programa. positivo: hay que mover la plataforma hacia atras
REFLECTANCE_PROBE_OCT_Z_PROGRAM_DISTANCE = -31.860  #mm programa positivo: hay que subir la sonda.

MOVEMENT_TO_CENTER_REFLECTANCE_PROBE = (18.20, 8.30, -15.6) #real mm
# === GLOBALS ===

scan_stop_requested = False
servers_connected = False
cnc_serial = None  # Global handler for serial port
spectrometer_client = None
spectrometer_client = None
oct_socket = None
oct_client = None
file_client = None
stepper_controller_client = None

# === Window lengths ===
SCAN_SIMPLE_X_REAL = 2.30 
SCAN_SIMPLE_Y_REAL = 2.30
SCAN_SIMPLE_Z_REAL = 2.88 #mm
SCAN_3D_X_REAL =  2.16
SCAN_3D_Y_REAL =  2.16

# === Scaxmax Config ===
SCAN_FOLDER = "testfolder"
SCAN_FILENAME = "oct"
SCAN_consecutive = 1

LABSCOPE_GATEWAY = "192.168.188.1"
oct_client_HOST = "192.168.188.200"
LABSCOPE_SERVER_HOST = "192.168.188.2"
LABSCOPE_SERVER_PORT = 12345

AVANTES_SERVER_HOST = "localhost"
AVANTES_SERVER_PORT = 11111

# === Shared State ===
position= {'x': 0.0, 'y': 0.0, 'z': 0.0, 'a': 0}
move_step_real = 1.0
pressed_keys = set()
stop_loop = False

#status vars
status_var = None
current_scope_status_var = None
oct_connection_status_var = None
spectrometer_connection_status_var = None
window = None

# === Flags ===
center_scope_flag = False
switch_scope_flag = False
home_flag = False

# === CONNECTION STATES ===

labscope_connection_state = LabscopeConnectionState.DISCONNECTED
avantes_connection_state = AvantesConnectionState.DISCONNECTED
current_scope_state = CurrentScopeState.OCT
        
# === Variables ===
dynamic_range_bottom_value = 0.37541

shared_state = {
    "rotation_angle": rotation_angle,
    "move_step_real": move_step_real,
    "stop_loop": stop_loop,
    "position": position,
    "home_flag": home_flag,
    "switch_scope_flag": switch_scope_flag,
    "center_scope_flag": center_scope_flag,
    "current_scope_state": current_scope_state,
    "pressed_keys": pressed_keys
}



# === Grid Movement State ===
class GridState(Enum):
    IDLE = auto()
    INITIAL_MOVE = auto()
    ROW_MOVEMENT = auto()
    COLUMN_ADVANCE = auto()
    SCANNING = auto()

grid_state = GridState.IDLE
grid_movement_event = Event()
grid_params = {'y_max': 0, 'n_max': 0, 'y_current': 0, 'x_current': 0, 
               'start_x': 0, 'start_y': 0, 'wait_start': 0}




from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

# Define a function to print the title in color and message
def print_message(title, message, color):
    # Map the color input to corresponding colorama color
    color_dict = {
        "red": Fore.RED,
        "green": Fore.GREEN,
        "yellow": Fore.YELLOW,
        "blue": Fore.BLUE,
        "magenta": Fore.MAGENTA,
        "cyan": Fore.CYAN,
        "white": Fore.WHITE,
        "black": Fore.BLACK
    }
    
    # Get the color from the dictionary, default to white if color is invalid
    color_code = color_dict.get(color.lower(), Fore.WHITE)
    color_code_white = color_dict.get("white", Fore.WHITE)
    
    # Print the message in the specified format
    print(f"{color_code}[{title}]{color_code_white} {message}")
    
def set_server_connection(host, port, timeout=1):
    """
    Intenta conectarse a un servidor TCP y retorna el socket abierto si tiene éxito.
    Si falla, retorna None.
    """
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((host, port))
        print(f"✅ Conectado a {host}:{port}")
        client_socket.settimeout(timeout)
        update_status()
        return client_socket  # socket abierto
    except socket.timeout:
        print(f"❌ Timeout al conectar con {host}:{port}")
    except Exception as e:
        print(f"❌ Error conectando a {host}:{port} -> {e}")

    return None

        

def close_sockets():
    global spectrometer_client, oct_client
    
    try:
        if spectrometer_client:
            spectrometer_client.close()
            print("Avantes socket cerrado.")
    except Exception as e:
        print("Error cerrando socket Avantes:", e)
    
    try:
        if oct_client:
            oct_client.close()
            print("Labscope socket cerrado.")
    except Exception as e:
        print("Error cerrando socket Labscope:", e)
        
def initialize_spectrometer_socket():
        global  avantes_connection_state, spectrometer_socket
        
        
        print_message(title= "SETUP", message = "Connecting to Spectrometer server", color = "green")
        
        # Labscope client initialization
        
        spectrometer_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)    
        spectrometer_socket.connect((LABSCOPE_SERVER_HOST, LABSCOPE_SERVER_PORT))
        
        
        if spectrometer_socket:
            
            avantes_connection_state = AvantesConnectionState.CONNECTED
        else:
           
            avantes_connection_state = AvantesConnectionState.DISCONNECTED
         
        
def initialize_spectrometer_client():
        global spectrometer_socket, spectrometer_client
        
        initialize_spectrometer_socket()
        
        spectrometer_client = Spectrometer_Client(spectrometer_socket)
            
        # Actualiza barra de estado o interfaz
        update_status()
        
        
def initialize_oct_socket(timeout = 120):
        global  labscope_connection_state, oct_socket
        
        
        print_message(title= "SETUP", message = "Connecting to OCT server", color = "green")
        
        # Labscope client initialization
        
        oct_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)    
        oct_socket.connect((LABSCOPE_SERVER_HOST, LABSCOPE_SERVER_PORT))
        oct_socket.settimeout(timeout)
        
        if oct_socket:
            
            labscope_connection_state = LabscopeConnectionState.CONNECTED
        else:
           
            labscope_connection_state = LabscopeConnectionState.DISCONNECTED
         
        
def initialize_oct_client():
        global oct_socket, oct_client
        
        initialize_oct_socket()
        
        oct_client = OCT_Client(oct_socket)
            
        # Actualiza barra de estado o interfaz
        update_status()
        

 
def initialize_spectrometer_client():
        global avantes_connection_state
        global spectrometer_client

        print_message(title= "SETUP", message = f"Connecting to spectrometer server", color = "green")
        
        spectrometer_client = set_server_connection(host=AVANTES_SERVER_HOST, port=AVANTES_SERVER_PORT, timeout=1)
        
        if spectrometer_client:    
            print_message(title= "SETUP", message = f"✅ Avantes conectado {spectrometer_client=}", color = "green")
            avantes_connection_state = AvantesConnectionState.CONNECTED
        else:
            print_message(title= "SETUP", message = f"❌ No se pudo conectar a Labscope", color = "red")
            avantes_connection_state = AvantesConnectionState.DISCONNECTED
        
        # initializa el cliente
        spectrometer_client = Spectrometer_Client(spectrometer_client)
        
        # Actualiza barra de estado o interfaz
        update_status() 
        
        #return client
        return spectrometer_client

def connect_stepper_controller_client():
    #initialize arduino controller
    try:
        stepper_controller_client = ArduinoStepperController(port='COM5')  # Modify COM port as needed
        
        # Initialize connections
        print_message(title= "SETUP", message = f"Connecting to stepper_controller_client", color = "green")
        stepper_controller_client.connect()
        print_message(title= "SETUP", message = f"Stepper connected successfully", color = "green")
        
        return stepper_controller_client
    except Exception as e:
        print_message(title= "SETUP", message = f"Unable to connect to stepper", color = "red")
        print(e)

def get_real_mm_step(digit):
    steps = {1: 0.05, 2: 0.10, 3: 0.30, 4: 1.00, 5: 2.00, 6: 5.00, 7: 10.00,
             8: 15.00, 9: 20.00, 0: 30.00}
    return steps.get(digit, 1.000)

def update_state_dictionary(callback = None):
    global move_step_real, rotation_angle,stop_loop, cnc_server, current_scope_state 
    global switch_scope_flag, center_scope_flag,home_flag, shared_state, pressed_keys
        
    if not callback:

        
        shared_state["rotation_angle"]= rotation_angle
        shared_state["move_step_real"]= move_step_real
        shared_state["stop_loop"]= stop_loop
        shared_state["position"]= cnc_server.position
        shared_state["home_flag"]= home_flag
        shared_state["switch_scope_flag"]= switch_scope_flag
        #print(f"{switch_scope_flag=}")
        shared_state["center_scope_flag"]= center_scope_flag
        shared_state["current_scope_state"]= current_scope_state
        #print(f"{current_scope_state=}")
        shared_state["pressed_keys"]= pressed_keys
        
        time.sleep(0.1)
        
    if callback:

        
        rotation_angle =  shared_state["rotation_angle"]
        move_step_real =  shared_state["move_step_real"]
        stop_loop =  shared_state["stop_loop"]
        cnc_server.position =  shared_state["position"]
        home_flag =  shared_state["home_flag"]
        switch_scope_flag =  shared_state["switch_scope_flag"]
        #print(f"{switch_scope_flag=}")
        center_scope_flag =  shared_state["center_scope_flag"]
        current_scope_state =  shared_state["current_scope_state"]
        #print(f"{current_scope_state=}")
        pressed_keys = shared_state["pressed_keys"]
        
        time.sleep(0.1)
        
        
def update_status(callback = None):
    
    global avantes_connection_state, labscope_connection_state,current_scope_state
    global cnc_server
    
    update_state_dictionary(callback)
    if status_var:
        delta_x, delta_y, delta_z, delta_a = cnc_server.get_relative_position()
        x, y, z, a = cnc_server.get_absolute_position_machine()
        xr, yr, zr,ar = cnc_server.get_absolute_position_real()
        
        status_text = f"""Pos (Relative, machine): X={delta_x:.3f} Y={delta_y:.3f} Z={delta_z:.3f} A={delta_a}\u00b0 | Step: {real_mm_to_program_units(move_step_real):.3f} mm
                          \n Pos (Absolute, Machine): X={x:.3f} Y={y:.3f} Z={z:.3f} A={a}\u00b0 | Step: {real_mm_to_program_units(move_step_real):.3f} mm 
                          \n Pos (Relative, Real) X={xr:.3f} Y={yr:.3f} Z={zr:.3f} A={position['a']}| | Step: {move_step_real} mm """
        if grid_state != GridState.IDLE:
            status_text += f" | Grid: {grid_state.name}"
        status_var.set(status_text)
        
        cnc_server.save_position(CURRENT_POSITION_FILE)
    #update connection status
    if oct_connection_status_var:
        status_text = f"|{labscope_connection_state} | "
        oct_connection_status_var.set(status_text)
        
    if spectrometer_connection_status_var:
        status_text = f" |{avantes_connection_state}   "
        spectrometer_connection_status_var.set(status_text)
        
    if current_scope_status_var:
        status_text = f"| {current_scope_state} | " 
        current_scope_status_var.set(status_text)

#request methods

def save_configuration_file(configuration_dict):
    
    readable_time = time.strftime("%Y-%m-%d %H.%M.%S", time.localtime())
    folder = configuration_dict["folder"]
    config_file_name = f"{folder}_{readable_time}.config"
    with open(config_file_name, "w") as f:
        f.write(f"{readable_time}\n")
        for key, value in configuration_dict.items():
           f.write(f"{key}\t{value}\n")

def request_3d_sweep_oct_scan(xi_max, yi_max, real_step_mm_x, real_step_mm_y ,  folder = "ts", filename = "simple3d", scan_type = "scan_3d"):
    global cnc_server, spectrometer_client, oct_client
    # Save initial position
    
    start_x = cnc_server.position['x']
    start_y = cnc_server.position['y']
    
    # Save configuration 
    focal_values = [50.0]
    focus_values = [50.0]
    start_time = time.time()
    alignment_scan_type = "horizontal"
    
    configuration_dict = {
                          "xi_max":xi_max,
                          "yi_max":yi_max,
                          "real_step_mm_x":real_step_mm_x,
                          "real_step_mm_y":real_step_mm_y,
                          "folder":folder,
                          "start_x":start_x,
                          "start_y":start_y,
                          "focal_values":focal_values,
                          "focus_values":focus_values,
                          "start_time":start_time,
                          "alignment_scan_type":alignment_scan_type,
                          "dynamic_range_bottom_value":dynamic_range_bottom_value,
                            }
    
    save_configuration_file(configuration_dict)
    #SETUP
   
    err_code = oct_client.send_oct_command_and_update_status("enter_folder_name", arguments = [folder])  
    err_code = oct_client.send_oct_command_and_update_status("set_alignment_scan_type", arguments = [alignment_scan_type])
    err_code = oct_client.send_oct_command_and_update_status("change_review_dynamic_range_bottom_value", arguments = [dynamic_range_bottom_value])
    
    #start labscope scan
    
    for yi in range(yi_max): 
        for xi in range(xi_max):
            consecutive = f"_{yi}-{xi}"
            print(f"{consecutive}")
            
            #send command 47.4 a 55.0
            
            
            for focus_counter in range(len(focus_values)):
                err_code  = oct_client.send_oct_command_and_update_status("change_focus_value", arguments = [focus_values[focus_counter]])
                for focal_value_counter in range(len(focal_values)):
                    
                    err_code  = oct_client.send_oct_command_and_update_status("change_focal_value", arguments = [focal_values[focal_value_counter]])
                    final_filename = f"{filename}{consecutive}_{focal_value_counter}_{focus_counter}"
                    err_code = oct_client.send_oct_command_and_update_status("enter_filename", arguments = [final_filename])
                    err_code = oct_client.send_oct_command_and_update_status(scan_type, arguments = [folder, final_filename])
                
            cnc_server.move_coordinate_x( steps = 1, program_mm = real_step_mm_x)
            
            #abort routine
            if scan_stop_requested:
                print("Escaneo abortado antes de iniciar Avantes.")
                # Regresar a la posición inicial
                cnc_server.return_to_start_position(x0 = start_x, y0 = start_y)
                return
        #when the loop ends, return to base x position
        cnc_server.move_coordinate_x( steps = -xi_max, program_mm = real_step_mm_x)
        #increase y_current
        cnc_server.move_coordinate_y( steps = +1, program_mm = real_step_mm_y)
    #whexmax the loop ends, return to base y position
    cnc_server.move_coordinate_y( steps = -yi_max, program_mm = real_step_mm_y)
    
    #end time
    end_time = time.time()
    
    #connection reset logic
    if err_code == OCTClientErrorCode.CONNECTION_RESET: 
        
        print("CONNECTION_RESET, stopping...")
        return
        
def request_3d_sweep_multi_frame_oct_scan(xi_max, yi_max, real_step_mm_x, real_step_mm_y ,  folder = "ts", filename = "simple3d"):
    global cnc_server, spectrometer_client, oct_client
    # Save initial position
    
    start_x = cnc_server.position['x']
    start_y = cnc_server.position['y']
    
    # Save configuration 
    focal_values = [44.19, 47.0, 51.0, 55.0]
    focus_values = [48.0, 50.0, 52.0]
    start_time = time.time()
    alignment_scan_type = "horizontal"
    
    configuration_dict = {
                          "xi_max":xi_max,
                          "yi_max":yi_max,
                          "real_step_mm_x":real_step_mm_x,
                          "real_step_mm_y":real_step_mm_y,
                          "folder":folder,
                          "start_x":start_x,
                          "start_y":start_y,
                          "focal_values":focal_values,
                          "focus_values":focus_values,
                          "start_time":start_time,
                          "alignment_scan_type":alignment_scan_type,
                          "dynamic_range_bottom_value":dynamic_range_bottom_value,
                            }
    
    save_configuration_file(configuration_dict)
    #SETUP
   
    err_code = oct_client.send_oct_command_and_update_status("enter_folder_name", arguments = [folder])  
    err_code = oct_client.send_oct_command_and_update_status("set_alignment_scan_type", arguments = [alignment_scan_type])
    err_code = oct_client.send_oct_command_and_update_status("change_review_dynamic_range_bottom_value", arguments = [dynamic_range_bottom_value])
    
    #start labscope scan
    
    for yi in range(yi_max): 
        for xi in range(xi_max):
            consecutive = f"_{yi}-{xi}"
            print(f"{consecutive}")
            
            #send command 47.4 a 55.0
            
            
            for focus_counter in range(3):
                err_code  = oct_client.send_oct_command_and_update_status("change_focus_value", arguments = [focus_values[focus_counter]])
                for focal_value_counter in range(3):
                    
                    err_code  = oct_client.send_oct_command_and_update_status("change_focal_value", arguments = [focal_values[focal_value_counter]])
                    final_filename = f"{filename}{consecutive}_{focal_value_counter}_{focus_counter}"
                    err_code = oct_client.send_oct_command_and_update_status("scan_3d", arguments = [folder, final_filename])
                
            cnc_server.move_coordinate_x( steps = 1, program_mm = real_step_mm_x)
            
            #abort routine
            if scan_stop_requested:
                print("Escaneo abortado antes de iniciar Avantes.")
                # Regresar a la posición inicial
                cnc_server.return_to_start_position(x0 = start_x, y0 = start_y)
                return
        #when the loop ends, return to base x position
        cnc_server.move_coordinate_x( steps = -xi_max, program_mm = real_step_mm_x)
        #increase y_current
        cnc_server.move_coordinate_y( steps = +1, program_mm = real_step_mm_y)
    #whexmax the loop ends, return to base y position
    cnc_server.move_coordinate_y( steps = -yi_max, program_mm = real_step_mm_y)
    
    #send folder back as zip file
    err_code = oct_client.send_oct_command_and_update_status("send_folder_to_client", arguments = [folder])
    
    #end time
    end_time = time.time()
    
    #connection reset logic
    if err_code == OCTClientErrorCode.CONNECTION_RESET: 
        
        print("CONNECTION_RESET, stopping...")
        return
        
def request_3d_sweep_single_frame_oct_scan(xi_max, yi_max, real_step_mm_x, real_step_mm_y ,  folder = "ts", filename = "simple3d"):
    global cnc_server, spectrometer_client, oct_client
    # Save initial position
    
    start_x = cnc_server.position['x']
    start_y = cnc_server.position['y']
    
    # Save configuration 
    focal_values = [44.19, 47.0, 51.0, 55.0]
    focus_values = [48.0, 50.0, 52.0]
    start_time = time.time()
    alignment_scan_type = "horizontal"
    
    configuration_dict = {
                          "xi_max":xi_max,
                          "yi_max":yi_max,
                          "real_step_mm_x":real_step_mm_x,
                          "real_step_mm_y":real_step_mm_y,
                          "folder":folder,
                          "start_x":start_x,
                          "start_y":start_y,
                          "focal_values":focal_values,
                          "focus_values":focus_values,
                          "start_time":start_time,
                          "alignment_scan_type":alignment_scan_type,
                          "dynamic_range_bottom_value":dynamic_range_bottom_value,
                            }
    
    save_configuration_file(configuration_dict)
    #SETUP
   
    err_code = oct_client.send_oct_command_and_update_status("enter_folder_name", arguments = [folder])  
    err_code = oct_client.send_oct_command_and_update_status("set_alignment_scan_type", arguments = [alignment_scan_type])
    err_code = oct_client.send_oct_command_and_update_status("change_review_dynamic_range_bottom_value", arguments = [dynamic_range_bottom_value])
    
    #start labscope scan
    
    for yi in range(yi_max): 
        for xi in range(xi_max):
            consecutive = f"_{yi}-{xi}"
            print(f"{consecutive}")
            
            #send command 47.4 a 55.0
            
            
            for focus_counter in range(3):
                err_code  = oct_client.send_oct_command_and_update_status("change_focus_value", arguments = [focus_values[focus_counter]])
                for focal_value_counter in range(3):
                    
                    err_code  = oct_client.send_oct_command_and_update_status("change_focal_value", arguments = [focal_values[focal_value_counter]])
                    final_filename = f"{filename}{consecutive}_{focal_value_counter}_{focus_counter}"
                    err_code = oct_client.send_oct_command_and_update_status("simple_scan", arguments = [folder, final_filename])
                
            cnc_server.move_coordinate_x( steps = 1, program_mm = real_step_mm_x)
            
            #abort routine
            if scan_stop_requested:
                print("Escaneo abortado antes de iniciar Avantes.")
                # Regresar a la posición inicial
                cnc_server.return_to_start_position(x0 = start_x, y0 = start_y)
                return
        #when the loop ends, return to base x position
        cnc_server.move_coordinate_x( steps = -xi_max, program_mm = real_step_mm_x)
        #increase y_current
        cnc_server.move_coordinate_y( steps = +1, program_mm = real_step_mm_y)
    #whexmax the loop ends, return to base y position
    cnc_server.move_coordinate_y( steps = -yi_max, program_mm = real_step_mm_y)
    
    #end time
    end_time = time.time()
    
    #connection reset logic
    if err_code == OCTClientErrorCode.CONNECTION_RESET: 
        
        print("CONNECTION_RESET, stopping...")
        return
        
        

def request_change_focal_value_oct(focal_value, client = None):
    global oct_client
    oct_client.send_oct_command_and_update_status("change_focal_value", arguments = [focal_value])
    
def request_3d_sweep_single_frame_dual_scan(xi_max, yi_max, real_step_mm_x, real_step_mm_y ,  folder = "ts", filename = "simple3d"):
    
    #request oct scan
    request_3d_sweep_single_frame_oct_scan(xi_max, yi_max, real_step_mm_x, real_step_mm_y ,  folder, filename)
    
    #start avantes scan
    request_3d_sweep_spectrometer_scan(xi_max, yi_max, real_step_mm_x, real_step_mm_y ,  folder , filename)
    
    
def request_3d_sweep_spectrometer_scan(xi_max, yi_max, real_step_mm_x, real_step_mm_y ,  folder = "ts", filename = "simple3d"):
    global scan_stop_requested, spectrometer_state, dark_state, reference_state, scan_mode, current_scope_state
    global cnc_server, spectrometer_client, oct_client
    
    scan_stop_requested = False  # reset at start

    # Guardar posición inicial
    start_x = cnc_server.position['x']
    start_y = cnc_server.position['y']
    
    #set reflectance probe in position
    if current_scope_state != CurrentScopeState.SPECTROMETER:
        cnc_server.center_reflectance_probe()
    
    time.sleep(6) #wait 6 seconds for CNC to finish
    
    print("get states")
    state = spectrometer_client.send_spectrometer_command_and_update_status(command="do_nothing", arguments=[])

    #if scan_mode != ScanMode.REFLECTANCE:
    #    print("set reflectance mode")
    #    send_spectrometer_command_and_update_status(command="reflectance_mode", arguments=None, client_socket = client_socket)
    
    print("Starting scan...")
    
    for yi in range(yi_max):
        if scan_stop_requested:
            print("Escaneo cancelado por el usuario.")
            break
        for xi in range(xi_max):
            if scan_stop_requested:
                print("Escaneo cancelado por el usuario.")
                break

            consecutive = f"_{yi}-{xi}"
            print(f"{consecutive}")
            
            filename_full = f"{get_today_str()}_{consecutive}"
            arguments = [folder, filename_full ]
            print(f"Requesting simple scan... {filename=}")
            state = spectrometer_client.send_spectrometer_command_and_update_status("save_spectra_ascii", arguments)

            time.sleep(0.5)
            
            if states:
                print(f"Response {states=}. States updated")
                cnc_server.move_coordinate_x( steps=1, program_mm=real_step_mm_x)
            else:
                time.sleep(0.5)
                print(f"Response {states=}. Not received?")
                cnc_server.move_coordinate_x( steps=1, program_mm=real_step_mm_x)
            #abort routine
            if scan_stop_requested:
                print("Escaneo abortado.")
                # Regresar a la posición inicial
                cnc_server.return_to_start_position(x0 = start_x, y0 = start_y)
                return

        cnc_server.move_coordinate_x( steps=-xi_max, program_mm=real_step_mm_x)
        cnc_server.move_coordinate_y( steps=1, program_mm=real_step_mm_y)

    cnc_server.move_coordinate_y( steps=-yi_max, program_mm=real_step_mm_y) 
    
    #center oct probe
    #set oct probe in position
    if current_scope_state == CurrentScopeState.SPECTROMETER:
        cnc_server.center_oct_probe()
        current_scope_state = CurrentScopeState.OCT


def initialize_file_client():
    print("[SETUP] Initializing file client...")
    root = tk.Tk()
    file_client = FileClientGUI(root)
    root.mainloop()
    
#create UI        
def create_ui():
    global status_var, spectrometer_connection_status_var, oct_connection_status_var,current_scope_status_var
    global stepper_controller_client, cnc_server
    global shared_state
    global window
    
    
    def setup():
        global stepper_controller_client, cnc_serial, cnc_server
        global spectrometer_client, oct_client
        print_message(title= "SETUP", message = "Starting setup", color = "green")
        
        #Connect
        
        cnc_serial = serial.Serial('COM7', 115200, timeout=1) #CNC
        
        stepper_controller_client = connect_stepper_controller_client()
        
        #initialize CNC 
        cnc_server = CNC_Server(cnc_serial, stepper_controller_client)
        
        #initialize oct_client and spectrometer_client
        
        try:
            initialize_oct_client()
        except Exception as e:
            print(e)
        initialize_spectrometer_client()
        
        #initialize_file_client
        threading.Thread(target=initialize_file_client, daemon=True).start()
        
    def on_closing():
        global current_scope_state, cnc_serial, stepper_controller_client
        print("Cerrando la aplicación...")
        
        #center probe
        if current_scope_state == CurrentScopeState.SPECTROMETER:
            current_scope_state = CurrentScopeState.OCT
            center_oct_probe(cnc_serial)
        
        #close sockets
        close_sockets()

        # Disconnect arduino after use 
        stepper_controller_client.disconnect()
        
        window.destroy()
        
        
    
    def exit_program():
        print("Saliendo de la aplicación...")
        close_sockets()
        window.destroy()

    def press_once(k):
        pressed_keys.add(k)
        window.after(100, lambda: pressed_keys.discard(k))

    def set_step(digit):
        global move_step_real
        move_step_real = get_real_mm_step(digit)
        update_status()


    def save_home_position_button():
        global cnc_server
        cnc_server.save_position(file = HOME_POSITION_FILE, reset = False)
        print("Saved home")
        update_status()
        
    def save_origin_position_button():
        global cnc_server
        cnc_server.save_position(file = ORIGIN_POSITION_FILE, reset = True)
        cnc_server.save_position(file = HOME_POSITION_FILE, reset = True)
        print("set origin")
        update_status()

    def home_machine_button():
        global home_flag
        home_flag = True
        update_status()
        
    def exit_program():
        global stop_loop
        stop_loop = True
        window.destroy()
        update_status()
    
            
    def switch_scope_button():
        global switch_scope_flag
        switch_scope_flag = True
        print(f"{switch_scope_flag=}")
        update_status()
        
        
    def center_scope_button():
        global current_scope_state, center_scope_flag
        center_scope_flag = True
        update_status()
        

    # Funciones para enviar escaneos
    def send_unique_dual_scan():
        global cnc_server
        global oct_client, spectrometer_client
        folder = folder_entry.get()
        filename = filename_entry.get()
        
        if oct_client:
            print(f"Solicitando oct single scan: {folder}/{filename}/{oct_client=}")
            oct_client.send_oct_command_and_update_status("enter_folder_name", arguments = [folder])  
            oct_client.send_oct_command_and_update_status("simple_scan", [folder, filename])
            
        if spectrometer_client:
            centered = False
            try:
                cnc_server.center_reflectance_probe()
                centered = True
                print(f"Midiendo espectro: {folder}/{filename}/{spectrometer_client=}")
                spectrometer_client.send_spectrometer_command_and_update_status("simple_scan", [folder, filename] )
                
            except Exception as e:
                if centered:
                    cnc_server.center_oct_probe()
        update_status()
        
    def send_oct_single_3d_scan():
        folder = folder_entry.get()
        filename = filename_entry.get()
        print(f"Solicitando scan_3d: {folder}/{filename}")
        ack = oct_client.send_oct_command_and_update_status("scan_3d", [folder, filename])
        print("Scan terminado")
        update_status()
        
    def send_spectrometer_3d_scan():
        global spectrometer_client
        #get folder, filename xmax ymax step x step y-yh
        folder = folder_entry.get()
        filename = filename_entry.get()
        ymax = int(ymax_entry.get())
        xmax = int(xmax_entry.get())
        step_x = float(grid_step_x_entry.get())
        step_y = float(grid_step_y_entry.get())
        
        folder = folder_entry.get()
        filename = filename_entry.get()
        print(f"Solicitando scan_3d: {folder}/{filename}")
        
        request_3d_sweep_spectrometer_scan(xi_max= xmax, yi_max = ymax, real_step_mm_x = step_x, real_step_mm_y = step_y ,
            folder = folder, filename = filename )
        update_status()
        
    def send_oct_3d_scan():
        #get folder, filename xmax ymax step x step y-yh
        folder = folder_entry.get()
        filename = filename_entry.get()
        ymax = int(ymax_entry.get())
        xmax = int(xmax_entry.get())
        step_x = float(grid_step_x_entry.get())
        step_y = float(grid_step_y_entry.get())
        
        folder = folder_entry.get()
        filename = filename_entry.get()
        print(f"Solicitando scan_3d: {folder}/{filename}")
        
        ack = request_3d_oct_scan(xi_max= xmax, yi_max = ymax, real_step_mm_x = step_x, real_step_mm_y = step_y ,
            folder = folder, filename = filename )
        update_status()
        
    def send_3d_sweep_single_frame_oct_scan():
        #get folder, filename xmax ymax step x step y-yh
        folder = folder_entry.get()
        filename = filename_entry.get()
        ymax = int(ymax_entry.get())
        xmax = int(xmax_entry.get())
        step_x = float(grid_step_x_entry.get())
        step_y = float(grid_step_y_entry.get())
        
        print(f"Requesting 3D sweep, oct with single frame scan: {folder}/{filename}")
        
        request_3d_sweep_single_frame_oct_scan(xi_max= xmax, yi_max = ymax, real_step_mm_x = step_x, real_step_mm_y = step_y ,
            folder = folder, filename = filename)
        update_status()
        
        
    def send_3d_sweep_single_frame_dual_scan():
        #get folder, filename xmax ymax step x step y-yh
        folder = folder_entry.get()
        filename = filename_entry.get()
        ymax = int(ymax_entry.get())
        xmax = int(xmax_entry.get())
        step_x = float(grid_step_x_entry.get())
        step_y = float(grid_step_y_entry.get())
        
        print(f"Solicitando barrido 3d con simple scan: {folder}/{filename}")
        
        request_3d_sweep_single_frame_dual_scan(xi_max= xmax, yi_max = ymax, real_step_mm_x = step_x, real_step_mm_y = step_y ,
            folder = folder, filename = filename )
        update_status()
    
    def send_3d_sweep_oct_scan():
        #get folder, filename xmax ymax step x step y-yh
        folder = folder_entry.get()
        filename = filename_entry.get()
        ymax = int(ymax_entry.get())
        xmax = int(xmax_entry.get())
        step_x = float(grid_step_x_entry.get())
        step_y = float(grid_step_y_entry.get())
        
        print(f"Solicitando barrido 3d con simple scan: {folder}/{filename}")
        
        request_3d_sweep_oct_scan(xi_max= xmax, yi_max = ymax, real_step_mm_x = step_x, real_step_mm_y = step_y ,
            folder = folder, filename = filename, scan_type = "scan_3d" )
        update_status()
        
    def send_toggle_oct_scan():
        global oct_client
        function = "start_scan"
        folder = folder_entry.get()
        filename = filename_entry.get()
        print(f"Solicitando {function}: {folder}/{filename}")
        oct_client.send_oct_command_and_update_status(function, [folder, filename])
        
        print(f"send_toggle_oct_scan")
        update_status()
    
    def send_change_focal_value_oct():
        focal_value = focal_value_entry.get()
        request_change_focal_value_oct(focal_value, client = oct_client)
        print(f"send_change_focal_value_oct")
        update_status()
        
    def connect_to_oct_server():
        initialize_oct_socket()
        initialize_oct_client()
        update_status()
        
    def connect_to_spectrometer_server():
        initialize_spectrometer_socket()
        initialize_spectrometer_client()
        update_status()
        
    def get_zip_folder():
        global oct_client
        folder = folder_entry.get()
        print("requesting folder as zip: ", folder)
        oct_client.send_oct_command_and_update_status("send_folder_to_client", arguments = [folder])
        
    from pynput.keyboard import Key

    def on_press(key):
        try:
            k = key.char.lower()
            pressed_keys.add(k)
            if k.isdigit():
                set_step(int(k))
        except AttributeError:
            if key == Key.up: pressed_keys.add('up')
            elif key == Key.down: pressed_keys.add('down')
            elif key == Key.left: pressed_keys.add('left')
            elif key == Key.right: pressed_keys.add('right')
            elif key == Key.esc: pressed_keys.add('esc')

    def on_release(key):
        try:
            k = key.char.lower()
            pressed_keys.discard(k)
        except AttributeError:
            if key == Key.up: pressed_keys.discard('up')
            elif key == Key.down: pressed_keys.discard('down')
            elif key == Key.left: pressed_keys.discard('left')
            elif key == Key.right: pressed_keys.discard('right')
            elif key == Key.esc: pressed_keys.discard('esc')
    #LOGIC
    #setup
    setup()
    
    print_message(title= "SETUP", message = f"Creating UI", color = "green")
    #create window
    window = tk.Tk()
    window.title("CNC Real-World Controller")
    window.protocol("WM_DELETE_WINDOW", on_closing)
    
    status_var = tk.StringVar()
    oct_connection_status_var = tk.StringVar()
    spectrometer_connection_status_var = tk.StringVar()
    current_scope_status_var = tk.StringVar()
   
    
    # Step size buttons
    for i in range(10):
        tk.Button(window, text=str(i), width=2, command=lambda i=i: set_step(i)).grid(row=4 + i // 5, column=i % 5)

    # Grid movement controls
    tk.Label(window, text="Grid Ymax:").grid(row=6, column=0)
    ymax_entry = tk.Entry(window, width=5)
    ymax_entry.grid(row=6, column=1)
    ymax_entry.insert(0, "3")
    
    tk.Label(window, text="Grid Xmax:").grid(row=6, column=2)
    xmax_entry = tk.Entry(window, width=5)
    xmax_entry.grid(row=6, column=3)
    xmax_entry.insert(0, "3")
    
    tk.Label(window, text="Focal value:").grid(row=6, column=4)
    focal_value_entry = tk.Entry(window, width=5)
    focal_value_entry.grid(row=6, column=5)
    focal_value_entry.insert(0, f"{50.0}")
    
    tk.Label(window, text="Grid Step X (mm):").grid(row=6, column=6)
    grid_step_x_entry = tk.Entry(window, width=5)
    grid_step_x_entry.grid(row=6, column=7)
    grid_step_x_entry.insert(0, f"{REAL_MM_PER_HORIZONTAL_FRAME}")
    #grid_step_x_entry.insert(0, f"0.1")
    
    tk.Label(window, text="Grid Step Y (mm):").grid(row=6, column=8)
    grid_step_y_entry = tk.Entry(window, width=5)
    grid_step_y_entry.grid(row=6, column=9)
    grid_step_y_entry.insert(0, "0.1")
    
    
    # Actions
    tk.Button(window, text="Save Home", command=save_home_position_button).grid(row=8, column=0)
    tk.Button(window, text="Go Home", command=home_machine_button).grid(row=8, column=1)
    tk.Button(window, text="Set Origin ", command=save_origin_position_button).grid(row=8, column=2)

    tk.Button(window, text="Exit", command=exit_program).grid(row=8, column=6)
    
    # Inputs para folder y filename
    tk.Label(window, text="Folder:").grid(row=11, column=0)
    folder_entry = tk.Entry(window, width=10)
    folder_entry.insert(0, f"{SCAN_FOLDER}-{get_today_str()}" )
    folder_entry.grid(row=11, column=1)

    tk.Label(window, text="Filename:").grid(row=12, column=0)
    filename_entry = tk.Entry(window, width=10)
    filename_entry.insert(0, SCAN_FILENAME)
    filename_entry.grid(row=12, column=1)
    
    # Botones de escaneo
    
    
    # Start Scan button
    
    start_scan_button = tk.Button(window, text="Toggle OCT Scan", command=send_toggle_oct_scan)
    start_scan_button.grid(row=13, column=1)
    switch_scope = tk.Button(window, text="Switch scope", command=switch_scope_button)
    switch_scope.grid(row=13, column=2)
    center_scope = tk.Button(window, text="Center scope", command=center_scope_button)
    center_scope.grid(row=13, column=3)
    tk.Label(window, textvariable=current_scope_status_var).grid(row=13, column=4, columnspan=3)
    focal_value_button = tk.Button(window, text="Change Focal value", command=send_change_focal_value_oct)
    focal_value_button.grid(row=13, column=7)
    
    
    simple_scan_button = tk.Button(window, text="Unique Dual-scan: (Mono Frame)", command=send_unique_dual_scan)
    simple_scan_button.grid(row=14, column=0)
    simple_scan_3d = tk.Button(window, text="Sweep Dual-Scan: (Mono Frame) ", command=send_3d_sweep_single_frame_dual_scan)
    simple_scan_3d.grid(row=14, column=1)
    unique_3d_oct_scan_button = tk.Button(window, text="Unique 3D OCT Scan: (Multiframe)", command=send_oct_single_3d_scan)
    unique_3d_oct_scan_button.grid(row=14, column=2)
    sweep_oct_scan_button = tk.Button(window, text="Sweep OCT Scan: (Mono Frame)", command=send_3d_sweep_single_frame_oct_scan)
    sweep_oct_scan_button.grid(row=14, column=3)
    sweep_spec_scan_button = tk.Button(window, text="Sweep Spectrometer Scan", command=send_spectrometer_3d_scan)
    sweep_spec_scan_button.grid(row=14, column=4)
    sweep_oct_scan_button_2 = tk.Button(window, text="Sweep OCT Scan: (Multi Frame)", command=send_3d_sweep_oct_scan)
    sweep_oct_scan_button_2.grid(row=14, column=5)
    

    #Connect button
    connect_button_oct = tk.Button(window, text="Connect to oct server", command=connect_to_oct_server)
    connect_button_oct.grid(row=15, column=0)
    connect_button_spectrometer = tk.Button(window, text="Connect to spectrometer server", command=connect_to_spectrometer_server)
    connect_button_spectrometer.grid(row=15, column=1)
    get_zip_button = tk.Button(window, text="Get folder as zip", command=get_zip_folder)
    get_zip_button.grid(row=15, column=2)
    # Status display
    tk.Label(window, textvariable=status_var).grid(row=16, column=0, columnspan=3)
    tk.Label(window, textvariable=oct_connection_status_var).grid(row=16, column=3, columnspan=3)
    tk.Label(window, textvariable=spectrometer_connection_status_var).grid(row=16, column=6, columnspan=3)
    
    
    
    # Start CNC loop thread
    cnc_server.start_daemon( update_state_dictionary, update_status, shared_state)
    
    #start listeners

    listener = keyboard.Listener(on_press=on_press,  on_release=on_release)
    listener.daemon = True
    listener.start()
    
    window.mainloop()

if __name__ == "__main__":
    create_ui()