import socket

# Server IP and port (change 'localhost' to the server's IP address)
host = 'localhost'  # Use the server's IP address (e.g., '192.168.1.2')
port = 12345        # Same port as the server is listening on

# Create a TCP socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect to the server
client_socket.connect((host, port))

# Send a message
message = "Hello from the client!"
client_socket.send(message.encode('utf-8'))

# Close the connection
client_socket.close()
