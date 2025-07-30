#avantes autoclicker client
from client_abstract import *
import socket
import time

AVANTES_SERVER_HOST = "localhost"
AVANTES_SERVER_PORT = 11111

from enum import Enum, auto


# === Grid Movement State ===
class SpectrometerState(Enum):
    IDLE = auto()
    SCANNING= auto()



class ReferenceState(Enum):
    NO_REFERENCE = auto()
    REFERENCE_OK= auto()


class DarkState(Enum):
    NO_DARK = auto()
    DARK_OK= auto()



class ScanMode(Enum):
    SCOPE = auto()
    TRANSMITTANCE = auto()
    REFLECTANCE = auto()
    ABSORPTANCE = auto()



class Spectrometer_Client(ClientAbstract):
    
    def __init__(self, socket):
        self.socket = socket
        self.instruction_id = 1
        self.queue = queue.Queue()
        
        self.scan_mode = ScanMode.SCOPE
        self.dark_state = DarkState.NO_DARK
        self.reference_state = ReferenceState.NO_REFERENCE
        self.spectrometer_state = SpectrometerState.IDLE
    
    #when ready add to avantes_autoclicker_client
    def send_spectrometer_command_and_update_status(self, command, arguments):
        try:
                client_socket = self.socket
                client_socket.settimeout(120)
                #create instruction
                message = self.create_avantes_message(command, arguments)
                #send instruction via reliable communication protocol
                print(f"sending spectrometer command {message=}")
                status_response = self.send_instruction(message)
                #update states
                self.update_states_spectrometer(status_response)
                return response
        except socket.timeout:
            print("Tiempo de espera agotado esperando ACK del servidor.")
            return False
        except Exception as e:
            print("Error en conexión con el servidor:", e)
            return False
        
        


    def update_states_spectrometer(self, response):
        # Check if the response is non-empty
        print(f" update_states_spectrometer {response=}")
        if not response:
            print("Empty response received.")
            return
        
        # If response is in bytes, decode it to a string
        if isinstance(response, bytes):
            response = response.decode('utf-8')  # You can choose the encoding type you need


        # Split the response string by a comma into a list
        response_list = response.split(",")
        print(f"{response_list=}")
        #first element is the total number of elements
        list_length = len(response_list)
        
        
        spectrometer_state_str, dark_state_str, reference_state_str, scan_mode_str = None,None,None,None
        
        # Assign variables from the response list
        if list_length >= 1:
            spectrometer_state_str = response_list[0]
        elif list_length >= 2:
            dark_state_str = response_list[1]
        elif list_length >= 3:
            reference_state_str = response_list[2]
        elif list_length >= 4:
            scan_mode_str = response_list[3]
        
        print(f" Processing states")
        # Process each state string and convert to the appropriate Enum value
        
        if  "IDLE" in spectrometer_state_str:
            spectrometer_state = SpectrometerState.IDLE
        elif "SCANNING" in spectrometer_state_str:
            spectrometer_state = SpectrometerState.SCANNING
        
        
        if  "DARK_OK" in dark_state_str:
            dark_state = DarkState.DARK_OK
        elif "NO_DARK" in dark_state_str :
            dark_state = DarkState.NO_DARK
        
        
        if "REFERENCE_OK" in reference_state_str:
            reference_state = ReferenceState.REFERENCE_OK
        elif "NO_REFERENCE" in reference_state_str:
            reference_state = ReferenceState.SCANNING
        
        
        if  "SCOPE" in scan_mode_str:
            scan_mode = ScanMode.SCOPE
        elif "TRANSMITTANCE" in scan_mode_str:
            scan_mode = ScanMode.TRANSMITTANCE
        elif "REFLECTANCE" in scan_mode_str:
            scan_mode = ScanMode.REFLECTANCE
        elif "ABSORPTANCE" in scan_mode_str:
            scan_mode = ScanMode.ABSORPTANCE
        
        # Print or handle the updated states
        print(f"Spectrometer state: {spectrometer_state}")
        print(f"Dark state: {dark_state}")
        print(f"Reference state: {reference_state}")
        print(f"Scan mode: {scan_mode}")
        
        
        return spectrometer_state, dark_state, reference_state, scan_mode 

    def get_spectrometer_states(self):
        states = f"{spectrometer_state},{dark_state},{reference_state},{scan_mode}"
        return states
        
    def create_avantes_message(self, command, arguments):
        message = f"{command}"
        for arg in arguments:
            message += f",{arg}"
        return message
    def avantes_send_instruction(self, client_socket, instruction, args):
        
        #arg string
        arg_string = str(args)
            
        # Formar el mensaje usando las variables
        message = f"{instruction},{get_spectrometer_states()}, {arg_string}"
        
        # Enviar el mensaje
        client_socket.sendall(message.encode('utf-8'))
        
        # Esperar la respuesta del servidor
        response = client_socket.recv(1024)  # Puedes ajustar el tamaño del búfer si esperas mensajes más largos
        
        return response 

def main():
    
    # Crear el socket TCP
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    try:
        while( True):
            
            print("Intentando conectar a", AVANTES_SERVER_HOST, AVANTES_SERVER_PORT)
            
            # Conectar con el servidor
            client_socket.connect((AVANTES_SERVER_HOST, AVANTES_SERVER_PORT))
            
            #create Spectrometer_Client object
            spectrometer_client = Spectrometer_Client(client_socket)
            
            instruction = "save_spectra"
            args = ["folder", "file"]
            
            while True:
                try:
                    # Formar el mensaje usando las variables
                    spectrometer_client.send_spectrometer_command_and_update_status(instruction, args )
                    time.sleep(10)
                except KeyboardInterrupt:
                    break
    except KeyboardInterrupt:
        print("finito")   
    finally:
        # Cerrar la conexión
        client_socket.close()
        
if __name__ == "__main__":
    main()
    
    
