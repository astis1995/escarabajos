import socket
import queue
import threading
from common import process_instruction
import re
import traceback

class ServerAbstract:
    def __init__(self, socket):
        self.sock = socket
        self.queue = queue.Queue()
        self.processed = set()
        self.running = False
        self.thread = None
        self.client_socket, self.client_address = None, None

    
    def set_socket(self, socket_obj):
        """Permite usar un socket externo en lugar del creado por defecto."""
        self.sock = socket_obj
        
    def _listen_loop(self):
        """Loop interno que se ejecuta en un hilo."""
        print("start listen loop")
        try:
            while self.running:
                try:
                    message = self.client_socket.recv(1024).decode('utf-8')
                    #print("message received:", message)
                    
                    if message.startswith("INSTRUCTION"):
                        
                        instruction_id = self.get_matching_re(message, r"INSTRUCTION (\d+):(.*)", 1)
                        instruction = self.get_matching_re(message, r"INSTRUCTION (\d+):(.*)", 2)
                        print("instruction received:", instruction, "instruction id: ", instruction_id)
                        
                        if instruction_id not in self.processed:
                            self.queue.put((instruction_id, instruction))

                        self.client_socket.sendall(f"ACK {instruction_id}".encode())
                        print(f"ACK {instruction_id} sent")
                    elif message.startswith("CONTINUE"):
                        if not self.queue.empty():
                            instruction_id, instruction = self.queue.get()
                            status = self.process_instruction(instruction_id, instruction)  # función inyectada
                            self.processed.add(instruction_id)
                            self.client_socket.sendall(f"STATUS: {status}".encode())
                        else:
                            status = self.get_status()
                            self.client_socket.sendall(f"STATUS: {status}".encode())

                except Exception as e:
                    print(f"[ERROR servidor]: {e}")
                    traceback.print_exc()
                    self.running = False
                    break
        except KeyboardInterrupt:
                    print(f"[Shutting down server]: {e}")
                    self.running = False
                    if self.client_socket:
                        self.client_socket.close()
                    
        
    def get_matching_re(self, text, pattern, n):

        match = re.match(pattern, text) 

        if match:
            full_match = match.group(0)   # Entire match: "$42.50"
            sub_match = match.group(n)    # Group n: "42.50"
            print("Full match:", full_match)
            print("Substring from regex group:", sub_match)
            return sub_match
        else:
            print("No match found.")
            
    def start(self):
        print("starting server")
        """Inicia el hilo de escucha."""
        self.running = True
        self.thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.thread.start()

    def retrieve_instruction(self):
        """Devuelve la siguiente instrucción pendiente como string, sin eliminarla."""
        if not self.queue.empty():
            instruction_id, instruction = self.queue.queue[0]
            return instruction_id,instruction
        return None
    
    def process_instruction(self, instruction_id, instruction):
        print(f"[DEFAULT] Procesando {instruction} con ID {instruction_id}")
        return f"Estado para {instruction}"

    
    def pop_instruction(self):
        """Elimina y devuelve la siguiente instrucción."""
        if not self.queue.empty():
            instruction_id, instruction = self.queue.get()
            return f"{instruction_id} {instruction}"
        return ""
    
    def accept(self):
        self.client_socket, self.client_address = self.sock.accept()
        
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        self.close()

    def close(self):
        self.client_socket.close()
        self.sock.close()

