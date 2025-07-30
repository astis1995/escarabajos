
import tkinter as tk
from threading import Thread, Event
import serial
import time
import os
import socket
from pynput import keyboard
from enum import Enum, auto
from spectrometer_client import *
from oct_client import *
from arduino_stepper_controller2 import ArduinoStepperController
import datetime
import time
import pyautogui
import numpy as np
from state_classes import *

#from test_CNC_UI4 import AvantesConnectionState, LabscopeConnectionState,CurrentScopeState

# === CNC Constants ===
PROGRAM_UNITS_PER_REAL_MM_X   = 1.631
PROGRAM_UNITS_PER_REAL_MM_Y   = 1.609
PROGRAM_UNITS_PER_REAL_MM_Z   = 1.593
PROGRAM_CLOCKWISE_DEGREES_PER_REAL_DEGREE = 1.0
PROGRAM_COUNTERCLOCKWISE_DEGREES_PER_REAL_DEGREE = 1.0
PROGRAM_UNITS_PER_REAL_MM = (PROGRAM_UNITS_PER_REAL_MM_X+PROGRAM_UNITS_PER_REAL_MM_Y+PROGRAM_UNITS_PER_REAL_MM_Z)/3
rotation_step = 45
CURRENT_POSITION_FILE = "current_position.txt"
HOME_POSITION_FILE = "home_position.txt"
ORIGIN_POSITION_FILE = "absolute_position_origin.txt"

PROGRAM_UNITS_PER_HORIZONTAL_FRAME = 4.322
REAL_MM_PER_HORIZONTAL_FRAME = PROGRAM_UNITS_PER_HORIZONTAL_FRAME/ PROGRAM_UNITS_PER_REAL_MM_X
calibration = True
X_MAX = 492.0 #mm programa
X_MIN = 0.0
Y_MAX = 221.0
Y_MIN = 0.0
Z_MAX = 70.0
Z_MIN = 0.0

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
LABSCOPE_CLIENT_HOST = "192.168.188.200"
LABSCOPE_SERVER_HOST = "192.168.188.2"
LABSCOPE_SERVER_PORT = 12345

AVANTES_SERVER_HOST = "localhost"
AVANTES_SERVER_PORT = 11111

CNC_SERVER_HOST = "localhost"
CNC_SERVER_PORT = 12000


# === Variables ===
angle_step_real = 1.0 #degrees
angle_variation_minimum = 1.0 #degrees


# === Grid Movement State ===
class CNCServerErrorCode(Enum):
    NO_ERROR = auto()
    TIMEOUT = auto()
    CONNECTION_RESET = auto()
    OTHER = auto()






# === Utility Functions ===

    
def get_today_str():
        return datetime.datetime.now().strftime("%Y-%m-%d")        


def real_mm_to_program_units(mm, coor = None): 
    if coor == "x":
        return round(mm * PROGRAM_UNITS_PER_REAL_MM_X, 5)
    elif coor == "y":
        return round(mm * PROGRAM_UNITS_PER_REAL_MM_Y, 5)
    elif coor == "z":
        return round(mm * PROGRAM_UNITS_PER_REAL_MM_Z, 5)
    else:
        return round(mm * PROGRAM_UNITS_PER_REAL_MM, 5)

def real_angle_to_program_units(real_angle, direction):
    program_angle = real_angle
    if direction == "clockwise":
        program_angle = real_angle * PROGRAM_CLOCKWISE_DEGREES_PER_REAL_DEGREE
        return program_angle
    elif direction == "counterclockwise":
        program_angle = real_angle * PROGRAM_COUNTERCLOCKWISE_DEGREES_PER_REAL_DEGREE
        return program_angle
def program_units_to_real_mm(units, coor = None): 
    if coor == "x":
        return round(units / PROGRAM_UNITS_PER_REAL_MM_X, 5)
    elif coor == "y":
        return round(units / PROGRAM_UNITS_PER_REAL_MM_Y, 5)
    elif coor == "z":
        return round(units / PROGRAM_UNITS_PER_REAL_MM_Z, 5)
    else:
        return round(units / PROGRAM_UNITS_PER_REAL_MM, 5)

def decode_unicode(line): 
    return line.decode('utf-8').strip()


#class definition

class CNC_Server():
    
    
    def __init__(self, ser, stepper_controller):
        self.ser = ser
        self.position = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'a': 0}
        self.stepper_controller = stepper_controller
        self.current_scope_state = CurrentScopeState.OCT
        
    def start_daemon(self, update_state_dictionary, update_status, shared_state):
        Thread(target=lambda: update_state_dictionary(), daemon=True).start()
        Thread(target=lambda: self.control_loop(update_status, self.ser, self.stepper_controller, shared_state), daemon=True).start()
        
    def send_wake_up(self): 
        self.ser.write(b'\r\n\r\n')
        time.sleep(2)
        self.ser.reset_input_buffer()


    def wait_for_movement_completion(self, manhattan_distance):
        time.sleep(0.5)
        run_counter = 0
        idle_counter = 0
        counter_max = int(manhattan_distance / 5)**2 + 1
        while True:
            self.ser.write(b'?\n')
            resp = decode_unicode(self.ser.readline())
            #print(f"{resp=}")
            if 'Idle' in resp or resp == 'ok': 
                idle_counter += 1
                time.sleep(0.1)
            if 'Run' in resp or resp == 'ok': 
                run_counter += 1
                time.sleep(0.1)
            else: 
                idle_counter += 0
                
            if run_counter + idle_counter > min(counter_max, 25): 
                #print(f"{run_counter=} {idle_counter=} {counter_max=} {manhattan_distance=}")
                
                break
        

    def send_grbl_command(self, command, manhattan_distance = 0):
        print(f">> {command}")
        self.ser.write((command + '\n').encode('utf-8'))
        self.wait_for_movement_completion( manhattan_distance)

    def send_move_command(self, dx=0, dy=0, dz=0):
        """Moves the probe in CNC units"""
        cmd = "G0"
        if dx != 0: cmd += f" X{dx}"
        if dy != 0: cmd += f" Y{dy}"
        if dz != 0: cmd += f" Z{dz}"
        manhattan_distance = np.abs(dx) + np.abs(dy) + np.abs(dz)
        self.send_grbl_command( cmd, manhattan_distance)

    def send_rotate_command(self, da):
        self.send_grbl_command( f"G0 A{da}")

    def save_position(self, file = CURRENT_POSITION_FILE, reset = False):
        #Saved as machine coordinates
        if reset:
            self.position['x'] = 0.0
            self.position['y'] = 0.0
            self.position['z'] = 0.0
            self.position['a'] = 0.0
            print("origin reset.")
        with open(file, "w") as f:
            f.write(f"{self.position['x']},{self.position['y']},{self.position['z']},{self.position['a']}")
    
    def move_to(self, x = None, y = None, z = None):
        
        if x:
            delta_x = x - self.position["x"]
            self.move_coordinate_x( 1, delta_x)
        if y: 
            delta_y = y - self.position["y"]
            self.move_coordinate_y( 1, delta_y)
        if z:
            delta_z = z - self.position["z"]
            self.move_coordinate_z( 1, delta_z)
        
    def return_to_start_position(self, x0 = None, y0 = None, z0 = None):
        self.move_to(x0, y0, z0)
        

    def load_position(self,file):
        
        
        try:
            if os.path.exists(file):
                with open(file, "r") as f:
                    file_content = f.read().strip().split(',')
                    #print(file_content)
                    
                    x, y, z, a = float(file_content[0]), float(file_content[1]), float(file_content[2]), float(file_content[3])
                    #print("position loaded.")
                    return x, y, z, a 
                    
            else:
                # Archivo no existe, crear uno con posición por defecto
                self.position.update(x=0.0, y=0.0, z=0.0, a=0.0)
                self.save_position(file)
                print("No saved position found.")
                return 0.0,0.0,0.0,0.0
        except:
            self.position.update(x=float(0), y=float(0), z=float(0), a=float(0.0))
            self.save_position(file)
            print("No saved position found.")
            return 0.0,0.0,0.0,0.0
        
    def save_home_position(self):
        self.save_position(HOME_POSITION_FILE)
        print("Home position saved.")


        
    def center_reflectance_probe(self):
        #set reflectance probe in position
        print("placing reflectance")
        x_program_units = REFLECTANCE_PROBE_OCT_X_PROGRAM_DISTANCE
        y_program_units = REFLECTANCE_PROBE_OCT_Y_PROGRAM_DISTANCE
        z_program_units = REFLECTANCE_PROBE_OCT_Z_PROGRAM_DISTANCE
        
        self.move_coordinate_x( steps=1, program_mm=x_program_units)  #go up in y
        self.move_coordinate_y( steps=1, program_mm=y_program_units)  #go up in y
        self.move_coordinate_z( steps=1, program_mm=z_program_units) #go down in z
        
        
    def center_oct_probe(self):
        print("placing oct reflectance")
        x_p_u = -1.0 *REFLECTANCE_PROBE_OCT_X_PROGRAM_DISTANCE
        y_p_u = -1.0 *REFLECTANCE_PROBE_OCT_Y_PROGRAM_DISTANCE
        z_p_u = -1.0 *REFLECTANCE_PROBE_OCT_Z_PROGRAM_DISTANCE
        
        self.move_coordinate_z( steps=1, program_mm=z_p_u) #go down in z
        self.move_coordinate_x( steps=1, program_mm=x_p_u)  #go up in y
        self.move_coordinate_y( steps=1, program_mm=y_p_u)  #go up in y
    
    
    def move_coordinate(self, steps, program_mm, coor):
        if steps == 0:
            return
        dl = steps * program_mm
        print(f"{dl=}")
        
        
        if coor == "x":
            new_tentative_position = self.position['x'] + dl
            max_limit, min_limit = X_MAX, X_MIN
            if ((new_tentative_position <= max_limit) and(new_tentative_position >= min_limit))or calibration:
                self.send_move_command( dx=dl)
                self.position['x'] = new_tentative_position
            else:
                print(f"New tentative position in {coor}: {new_tentative_position} is outbounds.({max_limit=},{min_limit=})")
        elif coor == "y":
            
            new_tentative_position = self.position['y'] + dl
            max_limit, min_limit = Y_MAX, Y_MIN
            if ((new_tentative_position <= max_limit) and(new_tentative_position >= min_limit))or calibration:
                self.send_move_command( dy=dl)
                self.position['y'] = new_tentative_position
            else:
                print(f"New tentative position in {coor}: {new_tentative_position} is outbounds.({max_limit=},{min_limit=})")
        elif coor == "z":
            
            new_tentative_position = self.position['z'] + dl
            max_limit, min_limit = Z_MAX, Z_MIN
            if ((new_tentative_position <= max_limit) and(new_tentative_position >= min_limit))or calibration:
                self.send_move_command( dz=dl)
                self.position['z'] = new_tentative_position
            else:
                print(f"New tentative position in {coor}: {new_tentative_position} is outbounds.({max_limit=},{min_limit=})")    
        elif coor == "a":
        
            new_tentative_position = position['a'] + dl
            max_limit, min_limit = 0, 360
            if ((new_tentative_position <= max_limit) and(new_tentative_position >= min_limit))or calibration:
                self.send_move_command( da=dl)
                self.position['a'] = new_tentative_position
            else:
                print(f"New tentative position in {coor}: {new_tentative_position} is outbounds.({max_limit=},{min_limit=})")
            
    def move_coordinate_x(self, steps, program_mm):
        """Moves the CNC head horizontally by `steps` steps of `program_mm` mm."""  
        self.move_coordinate( steps, program_mm, coor = "x")

        time.sleep(0.2)

    def move_coordinate_y(self, steps, program_mm):
        """Moves the CNC head in y by `steps` steps of `program_mm` mm."""
        self.move_coordinate( steps, program_mm, coor = "y")

        time.sleep(0.2)
        
    def move_coordinate_z(self, steps, program_mm):
        """Moves the CNC head in z by `steps` steps of `program_mm` mm."""
        self.move_coordinate( steps, program_mm, coor = "z")

        time.sleep(0.2)

    def move_coordinate_a(self, steps, angle):
        """Moves the CNC head in z by `steps` steps of `program_mm` mm."""
        self.stepper_controller.rotate(steps, angle)



    def get_absolute_position_machine(self):
        return self.load_position(CURRENT_POSITION_FILE)

    def get_absolute_position_real(self):
        x, y, z, a =  self.load_position(CURRENT_POSITION_FILE)
        xr = program_units_to_real_mm(x, "x")
        yr = program_units_to_real_mm(y, "y")
        zr = program_units_to_real_mm(z, "z")
        return xr, yr, zr, a
        
    
    def get_relative_position(self):
        #HOME POSITION
        xh, yh, zh, ah = self.load_position(HOME_POSITION_FILE) 
        #CURRENT_POSITION
        x, y, z, a = self.load_position(CURRENT_POSITION_FILE) 
        
        delta_x = x-xh
        delta_y = y-yh
        delta_z = z-zh
        delta_a = a-ah
        return delta_x,delta_y,delta_z,delta_a
    
    def home_machine(self):
        # === Mover a posición por defecto ===
        print("Moviendo a posición relativa por defecto: X=0mm, Y=0mm, Z=0mm")
        #RELATIVE POSITION
        xr, yr, zr, ar = self.get_relative_position()
        print("Relative position:", xr, yr, zr, ar)
        
        xa, ya, za, aa = self.get_absolute_position_machine()
        print("Absolute machine position:", xa, ya, za, aa)
        
        #ORIGIN_POSITION_FILE
        self.move_coordinate_x( steps=1, program_mm = -xr)
        self.move_coordinate_y( steps=1, program_mm = -yr)
        self.move_coordinate_z( steps=1, program_mm = -zr)
    
    
            
    
    def control_loop(self, update_status, ser, stepper_controller, shared_state):
        print("start control loop")
        

       
        def setup():
            
            self.send_wake_up()
            self.send_grbl_command( "G91")  # Relative positioning
            self.send_grbl_command( "G21")  # Use mm units
            # === Mover a posición por defecto ===
            #home_machine(ser)
            #connect to server
            
        
        def switch_scope(current_scope_state):
                      
            if current_scope_state == CurrentScopeState.OCT:
                
                return CurrentScopeState.SPECTROMETER
                
            elif current_scope_state == CurrentScopeState.SPECTROMETER:
                
                return CurrentScopeState.OCT
        
        def center_scope( current_scope_state):
             
            if current_scope_state == CurrentScopeState.OCT:
                            self.center_oct_probe()
                            
            elif current_scope_state == CurrentScopeState.SPECTROMETER:
                            self.center_reflectance_probe()
                        
        try:
            
                #setup
                setup()
                
                stop_loop = False
                

                while not stop_loop:
                    
                    #update flags
                    rotation_angle = shared_state["rotation_angle"] 
                    move_step_real = shared_state["move_step_real"]
                    stop_loop = shared_state["stop_loop"]
                    self.position = shared_state["position"]
                    home_flag  = shared_state["home_flag"]
                    switch_scope_flag = shared_state["switch_scope_flag"]
                    center_scope_flag = shared_state["center_scope_flag"]
                    current_scope_state = shared_state["current_scope_state"]
                    pressed_keys = shared_state["pressed_keys"]
                   
                    # Handle grid movement if active
                    #if grid_state != GridState.IDLE:
                    #    handle_grid_movement(ser, "scan_3d")
                    #    time.sleep(0.1)  # Small sleep to prevent busy SCANNING
                    #    continue
                    
                    #print(f"Control loop {current_scope_state=}")
                    #print(f" Control loop {switch_scope_flag=}")
                    
                    # Handle manual control
                    dx = dy = dz = 0.0
                    da = 0.0
                    
                    #convert move_step_real a unidades de programa
                    step_x = real_mm_to_program_units( move_step_real, "x")
                    step_y = real_mm_to_program_units( move_step_real, "y")
                    step_z = real_mm_to_program_units( move_step_real, "z")
                    step_a_clockwise = real_angle_to_program_units(angle_step_real, "clockwise")
                    step_a_counterclockwise = real_angle_to_program_units(angle_step_real, "counterclockwise") 
                    
                    if 'right' in pressed_keys:     dx += step_x
                    if 'left' in pressed_keys:      dx -= step_x
                    
                    
                    if 'up' in pressed_keys:        dy += step_y
                    if 'down' in pressed_keys:      dy -= step_y

                    if '+' in pressed_keys:         dz += step_z
                    if '}' in pressed_keys:         dz -= step_z
                    
                    if 'e' in pressed_keys:         da += step_a_clockwise #right hand rule
                    if 'q' in pressed_keys:         da -= step_a_counterclockwise #right hand rule
                    
                    
                    if home_flag:         
                        self.home_machine()
                        home_flag = False
                    
                    if switch_scope_flag:
                        current_scope_state = switch_scope(current_scope_state)
                        switch_scope_flag = False
                    
                    if center_scope_flag:         
                        center_scope( current_scope_state)
                        center_scope_flag = False
                        
                    if dx:
                        self.move_coordinate_x( steps = 1, program_mm = dx)
                    if dy:
                        self.move_coordinate_y( steps = 1, program_mm = dy)
                    if dz:
                        self.move_coordinate_z( steps = 1, program_mm = dz)
                    
                    if da > 0.0: #clockwise
                        print("da", da)
                        
                        self.move_coordinate_a( steps = 1, angle = da) 
                        new_angle = self.position['a'] + da
                        self.position['a'] = new_angle
                        
                    if da < 0.0: #cc
                        print("da", da)
                        self.move_coordinate_a( steps = 1, angle = da) 
                        new_angle = self.position['a'] + da
                        self.position['a'] = new_angle
                    
                    #update states
                    shared_state["rotation_angle"]= rotation_angle
                    shared_state["move_step_real"]= move_step_real
                    shared_state["stop_loop"]= stop_loop
                    shared_state["position"]= self.position
                    shared_state["home_flag"]= home_flag
                    shared_state["switch_scope_flag"]= switch_scope_flag
                    #print(f"{switch_scope_flag=}")
                    shared_state["center_scope_flag"]= center_scope_flag
                    shared_state["current_scope_state"]= current_scope_state
                    #print(f"{current_scope_state=}")
                    shared_state["pressed_keys"]= pressed_keys
                    
                    update_status(callback = True)
                    time.sleep(0.01)
                    
        except Exception as e:
            import traceback
            print("Control loop error:")
            traceback.print_exc()
        finally:
            pass



if __name__ == "__main__":
    pass
    #start_server() todo
    #just for testing: simple_scan(folder_name="testf", final_filename="final", wait_save = 8)
    #test: save_spectra_ascii(args= ["a", "b","c","d","e","f"])