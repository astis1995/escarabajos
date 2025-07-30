#oct clicker client
import socket
from enum import Enum, auto
import ast 
OCT_SERVER_HOST = "localhost"
OCT_SERVER_PORT = 11111


focal_value = 50

# === Grid Movement State ===
class ScanningState(Enum):
    IDLE = auto()
    SCANNING= auto()

scanning_state = ScanningState.IDLE

class CurrentTabState(Enum):
    MAIN = auto()
    REVIEW= auto()
    CONFIGURATION= auto()
    ADVANCED= auto()

current_tab_state = CurrentTabState.MAIN


#when ready add to avantes_autoclicker_client
def send_oct_command_and_get_state(command, arguments, client_socket):
    try:
            client_socket.settimeout(120)
            message = create_avantes_message(command, arguments)
            print(f"send oct command {message=}")
            client_socket.sendall(message.encode('utf-8'))
            ack_response = client_socket.recv(1024).decode('utf-8')
            print("Ack del servidor:", ack_response)
            #si se recibe ack
            if ack_response == "ack":
                #enviar continue
                continue_message = "continue"
                client_socket.sendall(continue_message.encode('utf-8'))
                #ahora recibe el estado
                status_response = client_socket.recv(1024).decode('utf-8')
                print("Status del servidor:", status_response)
                return status_response
            #si no, solo retorna la respuesta
            return response
    except socket.timeout:
        print("Tiempo de espera agotado esperando ACK del servidor.")
        return False
    except Exception as e:
        print("Error en conexión con el servidor:", e)
        return False
        
        



def update_states_oct(response):
    # Check if the response is non-empty
    print(f" update_states_oct {response=}")
    if not response:
        print("Empty response received.")
        return
    
    # If response is in bytes, decode it to a string
    if isinstance(response, bytes):
        response = response.decode('utf-8')  # You can choose the encoding type you need

    
    #convert response dict into a dictionary
    response_dict = ast.literal_eval(response)
    
    scanning = response_dict["scanning"]
    
    if "True" in scanning:
        scanning_state = ScanningState.SCANNING
    if "False" in scanning:
        scanning_state = ScanningState.IDLE
    
    focal_value = float(response_dict["focal_value"]
    
    
    current_tab = response_dict["current_tab_state"]
    
    if "main" in current_tab:
        current_tab_state = CurrentTabState.MAIN
    if "review" in current_tab:
        current_tab_state = CurrentTabState.REVIEW
    if "configuration" in current_tab:
        current_tab_state = CurrentTabState.CONFIGURATION
    if "advanced" in current_tab:
        current_tab_state = CurrentTabState.ADVANCED
        
    
    # Print or handle the updated states
    print(f"oct state: {scanning_state}")
    print(f"current_tab_state: {current_tab_state}")
    print(f": {focal_value}")

    

def get_oct_states():
    states = f"{scanning_state},{focal_value},{current_tab_state},{scan_mode}"
    return states
    

def main():
    
    # Crear el socket TCP
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    try:
        while( True):
            # Conectar con el servidor
            client_socket.connect((AVANTES_SERVER_HOST, AVANTES_SERVER_PORT))
            
            instruction = "save_spectra"
            args = f"test"
            
            # Formar el mensaje usando las variables
            response = send_spectrometer_command(client_socket, instruction, args )

            # Actualizar el estado
            update_states_oct(response)
            print(f"Respuesta del servidor: {response}, {oct_state},{dark_state},{reference_state},{scan_mode}")

    finally:
        # Cerrar la conexión
        client_socket.close()
        
if __name__ == "__main__":
    main()
    
    