#oct clicker client
import socket
import time
from enum import Enum, auto
import ast 
from client_abstract import *
from common import get_matching_re
import re
import json

OCT_SERVER_HOST = "192.168.188.2"
OCT_SERVER_PORT = 12345
MAX_RETRIES = 5

focal_value = 50

# === State ===
class AlignmentState(Enum):
    HORIZONTAL = auto()
    VERTICAL = auto()
    CROSS = auto()
    CIRCLE = auto()
    
class ScanningState(Enum):
    SCANNING = auto()
    NOT_SCANNING = auto()
    
class TabState(Enum):
    MAIN = auto()
    REVIEW = auto()
    CONFIGURATION = auto()
    ADVANCED = auto()
    
class OCTClientErrorCode(Enum):
    NO_ERROR = auto()
    TIMEOUT = auto()
    CONNECTION_RESET = auto()
    OTHER = auto()


# === Variables ===

focal_value = 50.0
focus_value = 50.0
power_value = 50.0
main_dynamic_range_bottom = 0.37541
review_dynamic_range_bottom = 0.37541
scanning_state = ScanningState.NOT_SCANNING
current_tab = TabState.MAIN
alignment = AlignmentState.HORIZONTAL
busy = False



class OCT_Client(ClientAbstract):
    
    def __init__(self, socket):
        self.socket = socket
        self.instruction_id = 1
        self.queue = queue.Queue()
        
        self.current_tab = TabState.MAIN
        self.scanning_state = ScanningState.NOT_SCANNING
        self.alignment = AlignmentState.HORIZONTAL
        
        self.focal_value = 50.0
        self.focus_value = 50.0
        self.power_value = 50.0
        self.main_dynamic_range_bottom = 0.37541
        self.review_dynamic_range_bottom = 0.37541
        self.dispersion_compensation_B = 0.0
        self.dispersion_compensation_C = 0.0
        
    def send_oct_command_and_update_status(self, command, arguments):
        try:
                client_socket = self.socket
                client_socket.settimeout(120)
                #create instruction
                message = f"{command}"
                for arg in arguments:
                    message += f",{arg}"
                #send instruction via reliable communication protocol
                status_response = self.send_instruction(message)
                #update states
                print("updating status")
                self.update_states_oct(status_response)
                #print("end")
                return OCTClientErrorCode.NO_ERROR #worked wll
        except socket.timeout:
            print("Tiempo de espera agotado esperando ACK del servidor.")
            return OCTClientErrorCode.TIMEOUT #timeout
        except ConnectionResetError:
            raise ConnectionResetError
            #return OCTClientErrorCode.CONNECTION_RESET #connection reset
        except Exception as e:
            import traceback
            print("Error en conexión con el servidor:", e)
            traceback.print_exc()
            return OCTClientErrorCode.OTHER #other error

    ENUM_REGISTRY = {
    "AlignmentState":AlignmentState,
    "TabState":TabState,
    "ScanningState":ScanningState,
    }
    def decode_status(self, data, enum_registry):
        if (isinstance(data,str) or isinstance(data,int) or isinstance(data,float)):
            return data
        
        def convert(obj):
            if "__enum__" in obj:
                enum_path = obj["__enum__"]
                enum_name, member_name = enum_path.split(".")
                enum_cls = enum_registry.get(enum_name)
                
                if enum_cls: 
                    return enum_cls[member_name]
                else:
                    raise ValueError(f"Unk enum: {enum_name}")
            return obj
        return json.loads(data.decode("utf-8"), object_hook = convert)

    def update_states_oct(self, response):
        # Check if the response is non-empty
        
        if "STATUS EMPTY" in response:
            print("Instruction sent and executed successfully")
            return
        if not response:
            print("Empty response received.")
            return
        #print("Response:", response)
        
        #get match for data
        pattern = r"STATUS: (.*)"
        status = get_matching_re( response, pattern, 1, debug = False)
        #print(type(status))
        status_b = ast.literal_eval(status)
        #print(type(status_b))
        
        response_dict = self.decode_status(status_b, OCT_Client.ENUM_REGISTRY)
        #print(type(response_dict) )
        #if it is a dict
        
        if "scanning" in response_dict.keys():
            self.scanning_state = ScanningState.SCANNING
        elif "alignment'" in response_dict.keys():
            self.alignment = response_dict["alignment"]
        elif "focal_value" in response_dict.keys():
            self.focal_value = response_dict["focal_value"]
        elif "current_tab" in response_dict.keys():
            self.current_tab = response_dict["current_tab"]
        elif "power_value" in response_dict.keys():
            self.power_value = response_dict["power_value"]
        elif "focus_value" in response_dict.keys():
            self.focus_value = response_dict["focus_value"]
        elif "dispersion_compensation_B" in response_dict.keys:
            self.dispersion_compensation_B = response_dict["dispersion_compensation_B"]
        elif "dispersion_compensation_C" in response_dict.keys:
            self.dispersion_compensation_C = response_dict["dispersion_compensation_C"]
        
            
        
        # Print or handle the updated states
        #print(f"""State: {self.scanning_state} {self.alignment}
        #        {self.focal_value}{self.current_tab}{self.power_value}{self.focus_value}
        #        {self.dispersion_compensation_B}{self.dispersion_compensation_C}""")


    

    def get_oct_states():
        states = f"{scanning_state},{focal_value},{current_tab},{scan_mode}"
        return states
    

def main():
    
    
    try:
        while True:
            print("Intentando conectar a", OCT_SERVER_HOST, OCT_SERVER_PORT)
    
            #initialize socket
            # Create a TCP/IP socket
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            
            client_socket.connect((OCT_SERVER_HOST, OCT_SERVER_PORT))
            
            #create OCT_Client object
            client = OCT_Client(client_socket)
            
            command = "start_scan"
            arguments = [f"test"]
            
            while True:
                try:
                    client.send_oct_command_and_update_status(command, arguments)
                    time.sleep(10)
                except KeyboardInterrupt:
                    break
    except KeyboardInterrupt:
        print("finito")
if __name__ == "__main__":
    main()
    
    