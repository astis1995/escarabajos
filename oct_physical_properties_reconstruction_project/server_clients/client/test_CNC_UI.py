import tkinter as tk
from threading import Thread, Event
import serial
import time
import os
from pynput import keyboard
from enum import Enum, auto

# === CNC Constants ===
PROGRAM_UNITS_PER_REAL_MM = 16 / 10
rotation_step = 45
POSITION_FILE = "saved_position.txt"

# === Shared State ===
position = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'a': 0}
move_step_real = 1.0
pressed_keys = set()
stop_loop = False
status_var = None
window = None

# === Grid Movement State ===
class GridState(Enum):
    IDLE = auto()
    INITIAL_MOVE = auto()
    ROW_MOVEMENT = auto()
    COLUMN_ADVANCE = auto()
    WAITING = auto()

grid_state = GridState.IDLE
grid_movement_event = Event()
grid_params = {'m_max': 0, 'n_max': 0, 'm_current': 0, 'n_current': 0, 
               'start_x': 0, 'start_y': 0, 'wait_start': 0}

# === Utility Functions ===
def get_real_mm_step(digit):
    steps = {1: 0.05, 2: 0.1, 3: 0.3, 4: 1.0, 5: 2.0, 6: 5.0, 7: 10.0,
             8: 15, 9: 20, 0: 30}
    return steps.get(digit, 1.00)

def real_mm_to_program_units(mm): 
    return round(mm * PROGRAM_UNITS_PER_REAL_MM, 4)

def program_units_to_real_mm(units): 
    return round(units / PROGRAM_UNITS_PER_REAL_MM, 4)

def send_wake_up(ser): 
    ser.write(b'\r\n\r\n')
    time.sleep(2)
    ser.reset_input_buffer()

def decode_unicode(line): 
    return line.decode('utf-8').strip()

def wait_for_movement_completion(ser):
    time.sleep(0.5)
    idle_counter = 0
    while True:
        ser.write(b'?\n')
        resp = decode_unicode(ser.readline())
        if 'Idle' in resp or resp == 'ok': 
            idle_counter += 1
        else: 
            idle_counter = 0
        if idle_counter > 2: 
            break
        time.sleep(0.2)

def send_grbl_command(ser, command):
    print(f">> {command}")
    ser.write((command + '\n').encode('utf-8'))
    wait_for_movement_completion(ser)

def send_move_command(ser, dx=0, dy=0, dz=0):
    cmd = "G0"
    if dx != 0: cmd += f" X{dx}"
    if dy != 0: cmd += f" Y{dy}"
    if dz != 0: cmd += f" Z{dz}"
    send_grbl_command(ser, cmd)

def send_rotate_command(ser, da):
    send_grbl_command(ser, f"G0 A{da}")

def save_position():
    with open(POSITION_FILE, "w") as f:
        f.write(f"{position['x']},{position['y']},{position['z']},{position['a']}")
    print("Position saved.")

def load_position():
    if os.path.exists(POSITION_FILE):
        with open(POSITION_FILE, "r") as f:
            x, y, z, a = f.read().strip().split(',')
            position.update(x=float(x), y=float(y), z=float(z), a=int(a))
            print("Position loaded.")
    else:
        print("No saved position found.")

def home_machine(ser):
    send_grbl_command(ser, "$H")
    position.update(x=0, y=0, z=0, a=0)
    print("Homed.")

def update_status():
    if status_var:
        status_text = f"Pos: X={position['x']:.1f} Y={position['y']:.1f} Z={position['z']:.1f} A={position['a']}° | Step: {move_step_real} mm"
        if grid_state != GridState.IDLE:
            status_text += f" | Grid: {grid_state.name}"
        status_var.set(status_text)

def start_grid_movement(m, n):
    global grid_state, grid_params
    if grid_state != GridState.IDLE:
        print("Grid movement already in progress")
        return
    
    grid_params = {
        'm_max': m-1,
        'n_max': n-1,
        'm_current': 0,
        'n_current': 0,
        'start_x': position['x'],
        'start_y': position['y'],
        'wait_start': 0
    }
    grid_state = GridState.INITIAL_MOVE
    grid_movement_event.set()
    print(f"Starting grid movement: {m}x{n}")

def stop_grid_movement():
    global grid_state
    if grid_state != GridState.IDLE:
        grid_state = GridState.IDLE
        grid_movement_event.clear()
        print("Grid movement stopped")

def handle_grid_movement(ser):
    global grid_state, position
    
    if grid_state == GridState.IDLE:
        return
    
    if grid_state == GridState.INITIAL_MOVE:
        # Initial move: -x'/2 and +y'/2
        send_move_command(ser, real_mm_to_program_units(-2.1 / 2), 
                         real_mm_to_program_units(6.4 / 2))
        position['x'] += -2.1 / 2
        position['y'] += 6.4 / 2
        grid_state = GridState.ROW_MOVEMENT
        update_status()
    
    elif grid_state == GridState.ROW_MOVEMENT:
        if grid_params['m_current'] < grid_params['m_max']:
            # Move in Y+ direction
            send_move_command(ser, 0, real_mm_to_program_units(6.4))
            position['y'] += 6.4
            grid_params['m_current'] += 1
            update_status()
            
            # Start waiting period (90 seconds)
            grid_params['wait_start'] = time.time()
            grid_state = GridState.WAITING
        else:
            # Move to next column or finish
            grid_state = GridState.COLUMN_ADVANCE
    
    elif grid_state == GridState.WAITING:
        elapsed = time.time() - grid_params['wait_start']
        if elapsed >= 90:  # 90 second wait completed
            grid_state = GridState.ROW_MOVEMENT
        else:
            print(f"Waiting... {90 - int(elapsed)}s remaining")
            time.sleep(1)  # Small sleep to prevent busy waiting
    
    elif grid_state == GridState.COLUMN_ADVANCE:
        if grid_params['n_current'] < grid_params['n_max']:
            # Reset Y to starting position
            send_move_command(ser, 0, real_mm_to_program_units(-6.4 * grid_params['m_max']))
            position['y'] -= 6.4 * grid_params['m_max']
            
            # Move one step in X- (next column)
            send_move_command(ser, real_mm_to_program_units(-2.1), 0)
            position['x'] -= 2.1
            
            grid_params['n_current'] += 1
            grid_params['m_current'] = 0
            grid_state = GridState.ROW_MOVEMENT
            update_status()
        else:
            # Grid movement complete
            print("Grid movement completed")
            grid_state = GridState.IDLE
            grid_movement_event.clear()

def control_loop(gui_update_callback):
    global move_step_real, stop_loop
    
    try:
        with serial.Serial('COM7', 115200, timeout=0.1) as ser:
            send_wake_up(ser)
            send_grbl_command(ser, "G91")  # Relative positioning
            send_grbl_command(ser, "G21")  # Use mm units
            load_position()
            
            while not stop_loop:
                # Handle grid movement if active
                if grid_state != GridState.IDLE:
                    handle_grid_movement(ser)
                    time.sleep(0.1)  # Small sleep to prevent busy waiting
                    continue
                
                # Handle manual control
                dx = dy = dz = 0
                da = 0
                step = real_mm_to_program_units(move_step_real)

                if 'w' in pressed_keys: dy += step
                if 's' in pressed_keys: dy -= step
                if 'a' in pressed_keys: dx -= step
                if 'd' in pressed_keys: dx += step
                if 'z' in pressed_keys: dz += step
                if 'c' in pressed_keys: dz -= step
                if 'q' in pressed_keys: da -= rotation_step
                if 'e' in pressed_keys: da += rotation_step

                if dx or dy or dz:
                    send_move_command(ser, dx, dy, dz)
                    position['x'] += program_units_to_real_mm(dx)
                    position['y'] += program_units_to_real_mm(dy)
                    position['z'] += program_units_to_real_mm(dz)

                if da:
                    send_rotate_command(ser, da)
                    position['a'] = (position['a'] + da) % 360

                gui_update_callback()
                time.sleep(0.05)
                
    except Exception as e:
        print("Control loop error:", e)
    finally:
        stop_grid_movement()

def create_ui():
    global status_var, window
    
    window = tk.Tk()
    window.title("CNC Real-World Controller")
    status_var = tk.StringVar()
    
    def press_once(k):
        pressed_keys.add(k)
        window.after(100, lambda: pressed_keys.discard(k))

    def set_step(digit):
        global move_step_real
        move_step_real = get_real_mm_step(digit)
        update_status()

    def home_machine_serial():
        try:
            with serial.Serial('COM7', 115200, timeout=0.1) as ser:
                send_wake_up(ser)
                home_machine(ser)
                update_status()
        except Exception as e:
            print("Error homing:", e)

    def exit_program():
        global stop_loop
        stop_loop = True
        window.destroy()

    def start_grid_from_ui():
        try:
            m = int(m_entry.get())
            n = int(n_entry.get())
            start_grid_movement(m, n)
        except ValueError:
            print("Invalid m or n value")

    # Movement buttons
    tk.Button(window, text="↑", width=6, command=lambda: press_once('w')).grid(row=0, column=1)
    tk.Button(window, text="←", width=6, command=lambda: press_once('a')).grid(row=1, column=0)
    tk.Button(window, text="→", width=6, command=lambda: press_once('d')).grid(row=1, column=2)
    tk.Button(window, text="↓", width=6, command=lambda: press_once('s')).grid(row=1, column=1)
    tk.Button(window, text="Z+", width=6, command=lambda: press_once('z')).grid(row=2, column=0)
    tk.Button(window, text="Z−", width=6, command=lambda: press_once('c')).grid(row=2, column=2)
    tk.Button(window, text="↺ Q", width=6, command=lambda: press_once('q')).grid(row=3, column=0)
    tk.Button(window, text="↻ E", width=6, command=lambda: press_once('e')).grid(row=3, column=2)
    tk.Button(window, text="Clear", width=6, command=lambda: pressed_keys.clear()).grid(row=3, column=1)

    # Step size buttons
    for i in range(10):
        tk.Button(window, text=str(i), width=2, command=lambda i=i: set_step(i)).grid(row=4 + i // 5, column=i % 5)

    # Grid movement controls
    tk.Label(window, text="Grid M:").grid(row=6, column=0)
    m_entry = tk.Entry(window, width=5)
    m_entry.grid(row=6, column=1)
    m_entry.insert(0, "2")
    
    tk.Label(window, text="Grid N:").grid(row=6, column=2)
    n_entry = tk.Entry(window, width=5)
    n_entry.grid(row=7, column=2)
    n_entry.insert(0, "7")
    
    tk.Button(window, text="Start Grid", command=start_grid_from_ui).grid(row=7, column=0)
    tk.Button(window, text="Stop Grid", command=stop_grid_movement).grid(row=7, column=1)

    # Actions
    tk.Button(window, text="Home (R)", command=home_machine_serial).grid(row=8, column=0)
    tk.Button(window, text="Save (P)", command=save_position).grid(row=8, column=1)
    tk.Button(window, text="Load (L)", command=load_position).grid(row=8, column=2)
    tk.Button(window, text="Exit", command=exit_program).grid(row=9, column=1)

    # Status display
    tk.Label(window, textvariable=status_var).grid(row=10, column=0, columnspan=3)

    # Start CNC loop thread
    Thread(target=lambda: control_loop(update_status), daemon=True).start()

    # Keyboard listener
    def on_press(key):
        try:
            k = key.char.lower()
            pressed_keys.add(k)
            if k.isdigit():
                set_step(int(k))
        except AttributeError:
            if key == keyboard.Key.esc:
                pressed_keys.add('esc')

    def on_release(key):
        try:
            k = key.char.lower()
            pressed_keys.discard(k)
        except AttributeError:
            if key == keyboard.Key.esc:
                pressed_keys.discard('esc')

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.daemon = True
    listener.start()

    window.mainloop()

if __name__ == "__main__":
    create_ui()