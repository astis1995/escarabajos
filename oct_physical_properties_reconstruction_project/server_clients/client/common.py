import socket
import time
import re
# Función para enviar una instrucción
def send_instruction(sock, instruction_id, action):
    message = f"INSTRUCTION {instruction_id}: {action}"
    sock.send(message.encode())

# Función para esperar un ACK
def wait_for_ack(sock, instruction_id):
    while True:
        message = sock.recv(1024).decode()
        if message == f"ACK {instruction_id}":
            print("ack received")
            return True
        elif message.startswith("ERROR"):
            return False

# Función para enviar un CONTINUE
def send_continue(sock):
    sock.send("CONTINUE".encode())

# Función para esperar un STATUS
def wait_for_status(sock):
    count = 0
    while(count < 120):
        status_message = sock.recv(1024).decode()
        if status_message:
            return status_message
        else:
            time.sleep(0.5)
            print("waiting for status", 120 - count)
            count += 1
    return status_message


def get_matching_re(text, pattern, n, debug = False):

        match = re.search(pattern, text)

        if match:
            full_match = match.group(0)   # Entire match: "$42.50"
            sub_match = match.group(n)    # Group 1: "42.50"
            if debug:
                print("Full match:", full_match)
                print(f"Substring from regex group {n}:", sub_match)
            return sub_match
        else:
            print("No match found.")