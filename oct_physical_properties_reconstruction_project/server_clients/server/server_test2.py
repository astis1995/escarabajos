import socket
import datetime
import os
from clicker_test import simple_scan, scan_3d

def get_today_str():
    return datetime.datetime.now().strftime("%Y-%m-%d")

def parse_message(message):
    parts = message.strip().split(',')
    # Fill missing parts with None for easier handling
    while len(parts) < 4:
        parts.append(None)
    scan_type, folder_name, filename, repetitions = parts

    scan_type = scan_type.strip() if scan_type else "simple_scan"
    folder_name = folder_name.strip() if folder_name else get_today_str()
    filename = filename.strip() if filename else get_today_str()
    repetitions = int(repetitions.strip()) if repetitions else 1

    return scan_type, folder_name, filename, repetitions

def get_scan_function(scan_type):
    if scan_type == "scan_3d":
        return scan_3d
    return simple_scan

def run_scan(scan_func, folder_name, filename_prefix, repetitions):
    counter = 1
    for i in range(repetitions):
        filename = f"{filename_prefix}{counter}"
        print(f"Running scan {i+1}/{repetitions} as {filename} in folder {folder_name}")
        scan_func(folder_name, filename)
        counter += 1

def start_server():
    host = '0.0.0.0'
    port = 12345
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(5)
    print(f"Server listening on port {port}...")

    client_socket, client_address = server_socket.accept()
    print(f"Connection from {client_address}")
    message = client_socket.recv(1024).decode('utf-8')
    print(f"Message received: {message}")
    
    scan_type, folder_name, filename_prefix, repetitions = parse_message(message)
    scan_func = get_scan_function(scan_type)
    run_scan(scan_func, folder_name, filename_prefix, repetitions)

    client_socket.close()
    server_socket.close()

if __name__ == "__main__":
    start_server()
