import socket
import time

# Función para enviar una instrucción
def send_instruction(sock, instruction_id, action):
    message = f"INSTRUCCION {instruction_id} {action}"
    sock.send(message.encode())

# Función para esperar un ACK
def wait_for_ack(sock, instruction_id):
    while True:
        message = sock.recv(1024).decode()
        if message == f"ACK {instruction_id}":
            return True
        elif message.startswith("ERROR"):
            return False

# Función para enviar un CONTINUE
def send_continue(sock):
    sock.send("CONTINUE".encode())

# Función para esperar un STATUS
def wait_for_status(sock):
    status_message = sock.recv(1024).decode()
    return status_message

# Función para procesar una instrucción en el servidor
def process_instruction(instruction_id, action):
    print(f"Procesando {action} con ID {instruction_id}")
    time.sleep(3)
    return f"Estado para {action}"
