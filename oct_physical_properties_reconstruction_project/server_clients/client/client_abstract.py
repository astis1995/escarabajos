import socket
import queue
from common import send_instruction, wait_for_ack, send_continue, wait_for_status

class ClientAbstract:
    def __init__(self,socket):
        self.socket = socket
        self.instruction_id = 1
        self.queue = queue.Queue()
    
    
    def set_socket(self, socket_obj):
        """Permite usar un socket externo en lugar del creado por defecto."""
        self.socket = socket_obj
        
    def send_instruction(self, instruction_str):
        """Agrega una instrucción en forma de string al buffer y la envía secuencialmente."""
        
        print("[SEND INSTRUCTION] adding instruction to queue", instruction_str)
        
        self.queue.put((self.instruction_id, instruction_str))

        while not self.queue.empty():
            current_id, instruction = self.queue.queue[0]
            
            print("[SEND INSTRUCTION] sending", instruction_str, "id", current_id)
            send_instruction(self.socket, current_id, instruction)
            
            print("waiting for ack")
            if wait_for_ack(self.socket, current_id):
                send_continue(self.socket)
                print("[SEND INSTRUCTION] continue sent. waiting for status")
                status = wait_for_status(self.socket)
                print("status received", status)
                self.queue.get()  # Confirmada y eliminada
                self.instruction_id += 1
                return status
            else:
                print(f"[SEND INSTRUCTION] Error with instruction {current_id}, retrying...")
        
    def close(self):
        self.socket.close()

