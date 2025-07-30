import socket

# Server settings
host = '0.0.0.0'  # Listen on all available interfaces (use 'localhost' for local connections)
port = 12345       # Port to listen on

# Create a TCP socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to the host and port
server_socket.bind((host, port))

# Start listening for incoming connections (maximum 5 connections)
server_socket.listen(5)
print(f"Server listening on port {port}...")

# Accept a connection
client_socket, client_address = server_socket.accept()
print(f"Connection from {client_address}")

# Receive and print the message from the client
message = client_socket.recv(1024).decode('utf-8')  # Receive up to 1024 bytes
print(f"Message received: {message}")

# Close the connection
client_socket.close()
server_socket.close()
