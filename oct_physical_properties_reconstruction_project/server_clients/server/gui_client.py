import socket
import threading
import tkinter as tk
from tkinter import filedialog, scrolledtext
from file_transfer import send_file, receive_file, prepare_file

SERVER_IP = '192.168.188.200'
PORT = 5001
DEST_FOLDER = 'client_received'

class ClientGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Cliente de Archivos")
        self.selected_path = None

        tk.Button(root, text="Seleccionar archivo/carpeta", command=self.select_path).pack(pady=5)
        tk.Button(root, text="Enviar al servidor", command=self.send_to_server).pack(pady=5)

        self.text_area = scrolledtext.ScrolledText(root, height=15, width=60)
        self.text_area.pack()

        threading.Thread(target=self.connect_to_server, daemon=True).start()

    def log(self, msg):
        self.text_area.insert(tk.END, msg + '\n')
        self.text_area.see(tk.END)

    def select_path(self):
        path = filedialog.askopenfilename()
        if not path:
            path = filedialog.askdirectory()
        if path:
            self.selected_path = path
            self.log(f"📂 Seleccionado: {path}")

    def connect_to_server(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.sock.connect((SERVER_IP, PORT))
            self.log(f"✅ Conectado a {SERVER_IP}:{PORT}")
            receive_file(self.sock, DEST_FOLDER, self.log)
        except Exception as e:
            self.log(f"❌ Error de conexión: {e}")

    def send_to_server(self):
        if not hasattr(self, 'sock'):
            self.log("❌ No conectado aún.")
            return
        if not self.selected_path:
            self.log("❌ No hay archivo/carpeta seleccionado.")
            return
        final_path = prepare_file(self.selected_path)
        send_file(self.sock, final_path, self.log)

if __name__ == '__main__':
    root = tk.Tk()
    app = ClientGUI(root)
    root.mainloop()
