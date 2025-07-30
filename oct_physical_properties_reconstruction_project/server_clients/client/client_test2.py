import socket

# Variables configurables
scan_type = "simple_scan"     # Puede ser: "simple_scan" o "scan_3d"
test_folder = "testfolder"
filename = "filex"
repetitions = 10

# Dirección IP y puerto del servidor
host = '192.168.13.175'
port = 12345

# Crear el socket TCP
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

try:
    # Conectar con el servidor
    client_socket.connect((host, port))

    # Formar el mensaje usando las variables
    message = f"{scan_type},{test_folder},{filename},{repetitions}"

    # Enviar el mensaje
    client_socket.sendall(message.encode('utf-8'))

    # Esperar la respuesta del servidor
    response = client_socket.recv(1024)  # Puedes ajustar el tamaño del búfer si esperas mensajes más largos
    print("Respuesta del servidor:", response.decode('utf-8'))

finally:
    # Cerrar la conexión
    client_socket.close()
