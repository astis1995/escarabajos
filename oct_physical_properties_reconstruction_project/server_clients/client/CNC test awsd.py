import serial
import time
from pynput import keyboard

#Aproximadamente 3.626 es el ancho de la pantalla
# These vars track pressed keys
pressed_keys = set()
move_step = 1  # Default step size in mm
SCREEN_WIDTH_MM = 3.626  # ancho de la pantalla en mm

def on_press(key):
    global move_step
    try:
        k = key.char.lower()
        pressed_keys.add(k)
        if k.isdigit():
            digit = int(k)

            if digit == 1:
                move_step = 0.1
            elif digit == 0:
                move_step = SCREEN_WIDTH_MM * 2  # 2×ancho para la tecla 0
            elif digit == 9:
                move_step = SCREEN_WIDTH_MM  # 2×ancho para la tecla 0
            elif digit == 2:
                move_step = SCREEN_WIDTH_MM * 1/10  #
            elif digit == 3:
                move_step = SCREEN_WIDTH_MM * 2/10  #
            elif digit == 4:
                move_step = 1 #mm 
            elif 4 <= digit <= 8:
                move_step = SCREEN_WIDTH_MM * (digit-1)/10 

            move_step = round(move_step, 3)
            print(f"Step size set to {move_step} mm")

    except AttributeError:
        if key == keyboard.Key.esc:
            pressed_keys.add('esc')



def on_release(key):
    try:
        pressed_keys.discard(key.char.lower())
    except AttributeError:
        if key == keyboard.Key.esc:
            pressed_keys.discard('esc')

# Start listener (non-blocking)
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

        # Accept 'Idle' status or 'ok' as idle indicators
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
    global move_step
    GRBL_com_port = 'COM7'
    BAUD_RATE = 115200

    # Open serial port
    ser = serial.Serial(GRBL_com_port, BAUD_RATE, timeout=0.1)
    send_wake_up(ser)

    print("Ready for manual control. Use WASD to move X/Y, Z/C for Z-axis, number keys 0-9 to set step size in mm.")
    print("Running in 0.2-second control loop.\n")
    print(f"Default step size: {move_step} mm")

    # Set to relative mode
    send_grbl_command(ser, "G91")  # Relative positioning
    send_grbl_command(ser, "G21")  # Set units to millimeters

    try:
        while True:
            dx = dy = dz = 0
            keys_pressed = []

            if 'w' in pressed_keys:
                dy += move_step
                keys_pressed.append('W')
            if 's' in pressed_keys:
                dy -= move_step
                keys_pressed.append('S')
            if 'd' in pressed_keys:
                dx -= move_step
                keys_pressed.append('A')
            if 'a' in pressed_keys:
                dx += move_step
                keys_pressed.append('D')
            if 'z' in pressed_keys:
                dz += move_step
                keys_pressed.append('Z')
            if 'c' in pressed_keys:
                dz -= move_step
                keys_pressed.append('C')
            if 'q' in pressed_keys or 'esc' in pressed_keys:
                print("Exiting control mode.")
                break

            if keys_pressed:
                print("Keys:", ", ".join(keys_pressed))
                command = "G0"
                if dx != 0:
                    command += f" X{dx}"
                if dy != 0:
                    command += f" Y{dy}"
                if dz != 0:
                    command += f" Z{dz}"
                send_grbl_command(ser, command)

            #print("Loop...")  # Prints every 0.2s
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("Keyboard Interrupt. Exiting.")

    send_grbl_command(ser, "G90")  # Absolute positioning

if __name__ == '__main__':
    main()


