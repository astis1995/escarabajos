import serial
import time
from pynput import keyboard

# Estos datos son en unidades de "programa"
SCREEN_WIDTH_PROGRAM_UNITS = 3.626  # Ancho de pantalla en unidades del programa
PROGRAM_UNITS_PER_REAL_MM = 16 / 10  # 16 mm del programa = 10 mm reales
SCREEN_WIDTH_REAL = SCREEN_WIDTH_PROGRAM_UNITS / PROGRAM_UNITS_PER_REAL_MM

pressed_keys = set()
move_step_real = 1  # mm reales

def get_real_mm_step(digit):
    """Devuelve el paso deseado en mm reales para una tecla numérica dada."""
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
        return SCREEN_WIDTH_REAL / 2
    elif digit == 9:
        return SCREEN_WIDTH_REAL
    elif digit == 0:
        return SCREEN_WIDTH_REAL * 2
    else:
        return 1.0  # valor por defecto

def real_mm_to_program_units(mm):
    """Convierte milímetros reales a unidades del programa."""
    return round(mm * PROGRAM_UNITS_PER_REAL_MM, 4)

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

def main():
    global move_step_real
    GRBL_com_port = 'COM7'
    BAUD_RATE = 115200

    ser = serial.Serial(GRBL_com_port, BAUD_RATE, timeout=0.1)
    send_wake_up(ser)

    print("Manual control. WASD for X/Y, Z/C for Z-axis, 0-9 to set step size in mm reales.")
    print(f"Pantalla: {SCREEN_WIDTH_REAL:.3f} mm reales de ancho.")
    send_grbl_command(ser, "G91")  # Modo relativo
    send_grbl_command(ser, "G21")  # mm

    try:
        while True:
            dx = dy = dz = 0
            keys_pressed = []

            move_step_program = real_mm_to_program_units(move_step_real)

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
            if 'q' in pressed_keys or 'esc' in pressed_keys:
                print("Exiting.")
                break

            if keys_pressed:
                print(f"Keys: {', '.join(keys_pressed)} | Move: {move_step_real:.2f} mm reales")
                command = "G0"
                if dx != 0:
                    command += f" X{dx}"
                if dy != 0:
                    command += f" Y{dy}"
                if dz != 0:
                    command += f" Z{dz}"
                send_grbl_command(ser, command)

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("Interrupted. Exiting.")

    send_grbl_command(ser, "G90")

if __name__ == '__main__':
    main()
