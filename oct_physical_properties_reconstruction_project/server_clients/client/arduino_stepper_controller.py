import serial
import time
import keyboard  # Import the keyboard library

class ArduinoStepperController:
    def __init__(self, port='COM5', baud_rate=9600):
        """
        Initializes the ArduinoStepperController to communicate with Arduino via serial.
        :param port: The COM port for Arduino (default is COM5, change as needed).
        :param baud_rate: The baud rate for serial communication (default is 9600).
        """
        self.port = port
        self.baud_rate = baud_rate
        self.arduino = None

    def connect(self):
        """
        Establishes a serial connection to the Arduino.
        """
        try:
            self.arduino = serial.Serial(self.port, self.baud_rate)
            time.sleep(2)  # Wait for Arduino to initialize
            print(f"Connected to Arduino on {self.port} at {self.baud_rate} baud.")
        except serial.SerialException as e:
            print(f"Error connecting to Arduino: {e}")
            self.arduino = None

    def disconnect(self):
        """
        Closes the serial connection to the Arduino.
        """
        if self.arduino and self.arduino.is_open:
            self.arduino.close()
            print("Disconnected from Arduino.")
        else:
            print("Arduino not connected or already disconnected.")

    def rotate(self, num_steps, angle):
        """
        Rotates the stepper motor by the given number of steps and angle.
        :param num_steps: Number of steps to move the stepper motor.
        :param angle: The angle by which to rotate the motor (positive for clockwise, negative for counterclockwise).
        """
        if not self.arduino:
            print("Arduino is not connected. Please connect to the Arduino first.")
            return

        # Send the number of steps and angle to Arduino
        try:
            self.arduino.write(f"{num_steps}\n".encode())  # Send the number of steps as a string
            self.arduino.write(f"{angle}\n".encode())      # Send the angle (positive or negative)
            time.sleep(0.015)  # Wait for the rotation to complete

            # Read and print the Arduino's response
            if self.arduino.in_waiting > 0:
                response = self.arduino.readline().decode('utf-8').strip()
                #print(response) debug
        except Exception as e:
            print(f"Error during rotation: {e}")

    def listen_for_keys(self):
        """
        Listens for keypresses ('Q' for counterclockwise, 'E' for clockwise).
        """
        #print("Press 'Q' to rotate counterclockwise, 'E' to rotate clockwise.")
        
        while True:
            if keyboard.is_pressed('q'):  # If the 'Q' key is pressed
                self.rotate(10, -0.5)  # Rotate counterclockwise by 0.5°
                time.sleep(0.5)  # Delay to prevent rapid firing
            elif keyboard.is_pressed('e'):  # If the 'E' key is pressed
                self.rotate(10, 0.5)  # Rotate clockwise by 0.5°
                time.sleep(0.5)  # Delay to prevent rapid firing

# Example usage:
if __name__ == "__main__":
    # Create an instance of the ArduinoStepperController
    controller = ArduinoStepperController(port='COM5')  # Modify COM port as needed
    
    # Connect to the Arduino
    controller.connect()

    # Listen for key presses
    controller.listen_for_keys()

    # Disconnect after use (this won't be reached unless the loop is manually stopped)
    controller.disconnect()
