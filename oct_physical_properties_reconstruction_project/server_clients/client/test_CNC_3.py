import serial
import time
from pynput import keyboard
import os

# Real-world axis ranges in mm
X_AXIS_RANGE_MM = 340.0
Y_AXIS_RANGE_MM = 200.0
Z_AXIS_RANGE_MM = 80.0

# Offsets to center the coordinate system (so 0,0,0 is mid-travel)
X_OFFSET = -X_AXIS_RANGE_MM / 2  # -170.0
Y_OFFSET = -Y_AXIS_RANGE_MM / 2  # -100.0
Z_OFFSET = -Z_AXIS_RANGE_MM / 2  # -40.0

# Conversion factor (program units to real mm)
SCREEN_WIDTH_PROGRAM_UNITS = 3.626  # From your original program units
PROGRAM_UNITS_PER_REAL_MM = 16 / 10  # 16 program units = 10 real mm

# Position tracking in real mm (centered coordinates)
position = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'a': 0}  # 'a' for rotation in degrees

pressed_keys = set()
move_step_real = 1  # Default step size in mm real
rotation_step = 45  # degrees per rotation step

POSITION_FILE = "saved_position.txt"

def get_real_mm_step(digit):
    """Return movement step in mm real for key digit."""
    if digit == 1:
        return 0.1
    elif digit == 2:
        return 0.3
    elif digit == 3:
        return 0.5
    elif digit == 4:
        return 1.0
    elif digit == 5:
        return 2.0
    elif digit == 6:
        return 5.0
    elif digit == 7:
        return 10.0
    elif digit == 8:
        return Y_AXIS_RANGE_MM / 2  # half screen height (you can adjust)
    elif digit == 9:
        return Y_AXIS_RANGE_MM       # full screen height
    elif digit == 0:
        return Y_AXIS_RANGE_MM * 2   # twice screen height
    else:
        return 1.0

def real_mm_to_program_units(mm):
    """Convert mm real to program units."""
    return round(mm * PROGRAM_UNITS_PER_REAL_MM, 4)

def program_units_to_real_mm(units):
    """Convert program units back to real mm."""
    return round(units / PROGRAM_UNITS_PER_REAL_MM, 4)

def apply_offsets(real_x, real_y, real_z):
    """Apply offset to convert from absolute real mm position to centered coordinate system."""
    return (real_x + X_OFFSET, real_y + Y_OFFSET, real_z + Z_OFFSET)

def remove_offsets(centered_x, centered_y, centered_z):
    """Convert centered coordinates back to absolute real mm."""
    return (centered_x - X_OFFSET, centered_y - Y_OFFSET, centered_z - Z_OFFSET)

def on_press(key):
    global move_step_real
    try:
        k = key.char.lower()
        pressed_keys.add(k)
        if k.isdigit():
            digit = int(k)
            move_step_real = get_real_mm_step(digit)
            print(f"Step size set to {move_step_real:.3f} mm reales")
    except AttributeError:
        if key == keyboard.Key.esc:
            pressed_keys.add('esc')
        elif key == keyboard.Key.space:
            # You can define a key for save/load or other commands here if desired
            pass

def on_release(key):
    try:
        pressed_keys.discard(key.char.lower())
    except AttributeError:
        if key == keyboard.Key.esc:
            pressed_keys.discard('esc')

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

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
        grbl_out = ser.readline()
        grbl_response = decode_unicode(grbl_out)
        if 'Idle' in grbl_response or grbl_response == 'ok':
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

def save_position():
    """Save the current real mm centered position and rotation to a file."""
    with open(POSITION_FILE, "w") as f:
        f.write(f"{position['x']},{position['y']},{position['z']},{position['a']}\n")
    print(f"Position saved to {POSITION_FILE}")

def load_position():
    """Load position and rotation from file if exists."""
    if os.path.exists(POSITION_FILE):
        with open(POSITION_FILE, "r") as f:
            line = f.readline().strip()
            if line:
                x_str, y_str, z_str, a_str = line.split(',')
                position['x'] = float(x_str)
                position['y'] = float(y_str)
                position['z'] = float(z_str)
                position['a'] = int(a_str)
                print(f"Position loaded: x={position['x']}, y={position['y']}, z={position['z']}, a={position['a']}°")
    else:
        print("No saved position file found.")

def send_move_command(ser, dx=0, dy=0, dz=0):
    """Send movement command with given program units deltas."""
    command = "G0"
    if dx != 0:
        command += f" X{dx}"
    if dy != 0:
        command += f" Y{dy}"
    if dz != 0:
        command += f" Z{dz}"
    send_grbl_command(ser, command)

def send_rotate_command(ser, da):
    """Send rotation command (A axis) with relative degrees."""
    command = f"G0 A{da}"
    send_grbl_command(ser, command)

def home_machine(ser):
    """Send homing command and reset position."""
    print("Sending homing command...")
    send_grbl_command(ser, "$H")  # GRBL homing command
    # After homing, position resets to machine zero (assumed to be min corner)
    # Update internal position to center (0,0,0) based on offsets
    position['x'] = 0.0
    position['y'] = 0.0
    position['z'] = 0.0
    position['a'] = 0
    print("Homing done. Position reset to center (0,0,0).")

def main():
    global move_step_real
    GRBL_com_port = 'COM7'
    BAUD_RATE = 115200

    ser = serial.Serial(GRBL_com_port, BAUD_RATE, timeout=0.1)
    send_wake_up(ser)

    print("Manual control active.")
    print("WASD: move X/Y, Z/C: move Z, Q/E: rotate A axis (45° steps), R: home,")
    print("0-9: set movement step size (in real mm).")
    print(f"Coordinate system centered at mid-travel:")
    print(f"  X: {X_OFFSET} mm to {X_OFFSET + X_AXIS_RANGE_MM} mm")
    print(f"  Y: {Y_OFFSET} mm to {Y_OFFSET + Y_AXIS_RANGE_MM} mm")
    print(f"  Z: {Z_OFFSET} mm to {Z_OFFSET + Z_AXIS_RANGE_MM} mm")

    # Set relative mode and mm units
    send_grbl_command(ser, "G91")  # relative positioning
    send_grbl_command(ser, "G21")  # mm units

    load_position()

    try:
        while True:
            dx = dy = dz = 0
            da = 0
            keys_pressed = []

            move_step_program = real_mm_to_program_units(move_step_real)

            # Movement controls
            if 'w' in pressed_keys:
                dy += move_step_program
                keys_pressed.append('W')
            if 's' in pressed_keys:
                dy -= move_step_program
                keys_pressed.append('S')
            if 'a' in pressed_keys:
                dx -= move_step_program
                keys_pressed.append('A')
            if 'd' in pressed_keys:
                dx += move_step_program
                keys_pressed.append('D')
            if 'z' in pressed_keys:
                dz += move_step_program
                keys_pressed.append('Z')
            if 'c' in pressed_keys:
                dz -= move_step_program
                keys_pressed.append('C')

            # Rotation controls
            if 'q' in pressed_keys:
                da -= rotation_step
                keys_pressed.append('Q')
            if 'e' in pressed_keys:
                da += rotation_step
                keys_pressed.append('E')

            # Auto home
            if 'r' in pressed_keys:
                home_machine(ser)
                pressed_keys.discard('r')

            # Save position
            if 'p' in pressed_keys:
                save_position()
                pressed_keys.discard('p')

            # Load position
            if 'l' in pressed_keys:
                load_position()
                pressed_keys.discard('l')

            # Quit
            if 'esc' in pressed_keys or 'ñ' in pressed_keys and not ('e' in pressed_keys or 'w' in pressed_keys or 'a' in pressed_keys or 's' in pressed_keys or 'z' in pressed_keys or 'c' in pressed_keys):
                print("Exiting.")
                break

            # Send movement command if needed
            if dx != 0 or dy != 0 or dz != 0:
                send_move_command(ser, dx, dy, dz)
                # Update position (convert program units back to real mm)
                position['x'] += program_units_to_real_mm(dx)
                position['y'] += program_units_to_real_mm(dy)
                position['z'] += program_units_to_real_mm(dz)

            # Send rotation command if needed
            if da != 0:
                send_rotate_command(ser, da)
                position['a'] = (position['a'] + da) % 360

            if keys_pressed:
                print(f"Keys: {', '.join(keys_pressed)} | Step: {move_step_real:.2f} mm reales")
                print(f"Position (centered): X={position['x']:.2f} mm, Y={position['y']:.2f} mm, Z={position['z']:.2f} mm, A={position['a']}°")

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("Interrupted. Exiting.")

    send_grbl_command(ser, "G90")  # back to absolute mode

if __name__ == '__main__':
    main()
