import socket
import threading
import tkinter as tk
from tkinter import filedialog, scrolledtext
from file_transfer import send_file, receive_file, prepare_file

PORT = 5001
DEST_FOLDER = 'server_received'

class ServerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Servidor de Archivos")
        self.selected_path = None

        tk.Button(root, text="Seleccionar archivo/carpeta", command=self.select_path).pack(pady=5)
        tk.Button(root, text="Enviar al cliente", command=self.send_to_client).pack(pady=5)

        self.text_area = scrolledtext.ScrolledText(root, height=15, width=60)
        self.text_area.pack()

        threading.Thread(target=self.wait_for_connection, daemon=True).start()

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

    def wait_for_connection(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_sock:
            server_sock.bind(('', PORT))
            server_sock.listen(1)
            self.log(f"🔌 Esperando conexión en puerto {PORT}...")
            self.conn, self.addr = server_sock.accept()
            self.log(f"✅ Conectado desde {self.addr}")
            receive_file(self.conn, DEST_FOLDER, self.log)

    def send_to_client(self):
        if not hasattr(self, 'conn'):
            self.log("❌ Esperando conexión del cliente...")
            return
        if not self.selected_path:
            self.log("❌ No hay archivo/carpeta seleccionado.")
            return
        final_path = prepare_file(self.selected_path)
        send_file(self.conn, final_path, self.log)
    
    
        
if __name__ == '__main__':
    root = tk.Tk()
    app = ServerGUI(root)
    root.mainloop()
