import tkinter as tk
from tkinter import filedialog, messagebox
import socket
import os
import threading

PORT = 5001  # Must match the server's port

def send_files(server_ip, source_folder, status_label):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((server_ip, PORT))
            status_label.config(text=f"Connected to {server_ip}")

            for filename in os.listdir(source_folder):
                filepath = os.path.join(source_folder, filename)
                if os.path.isfile(filepath):
                    filesize = os.path.getsize(filepath)
                    header = f"{filename},{filesize}"
                    s.sendall(header.encode('utf-8'))

                    with open(filepath, 'rb') as f:
                        while chunk := f.read(4096):
                            s.sendall(chunk)
            s.sendall(b'DONE')
            status_label.config(text="All files sent successfully.")
    except Exception as e:
        status_label.config(text=f"Error: {e}")
        messagebox.showerror("Transfer Error", str(e))

def start_transfer(ip_entry, folder_entry, status_label):
    server_ip = ip_entry.get().strip()
    source_folder = folder_entry.get().strip()

    if not server_ip or not source_folder:
        messagebox.showwarning("Input Missing", "Please fill in both fields.")
        return
    if not os.path.isdir(source_folder):
        messagebox.showerror("Invalid Path", "Source folder does not exist.")
        return

    # Use a thread to keep GUI responsive
    threading.Thread(target=send_files, args=(server_ip, source_folder, status_label), daemon=True).start()

def browse_folder(entry_widget):
    folder_selected = filedialog.askdirectory()
    if folder_selected:
        entry_widget.delete(0, tk.END)
        entry_widget.insert(0, folder_selected)

def build_gui():
    root = tk.Tk()
    root.title("File Sender")

    tk.Label(root, text="Destination IP:").grid(row=0, column=0, sticky='e', padx=5, pady=5)
    ip_entry = tk.Entry(root, width=40)
    ip_entry.grid(row=0, column=1, columnspan=2, padx=5, pady=5)

    tk.Label(root, text="Source Folder:").grid(row=1, column=0, sticky='e', padx=5, pady=5)
    folder_entry = tk.Entry(root, width=40)
    folder_entry.grid(row=1, column=1, padx=5, pady=5)
    browse_button = tk.Button(root, text="Browse", command=lambda: browse_folder(folder_entry))
    browse_button.grid(row=1, column=2, padx=5, pady=5)

    status_label = tk.Label(root, text="", fg="blue")
    status_label.grid(row=3, column=0, columnspan=3, padx=5, pady=10)

    send_button = tk.Button(root, text="Send Files", command=lambda: start_transfer(ip_entry, folder_entry, status_label))
    send_button.grid(row=2, column=1, pady=10)

    root.mainloop()

if __name__ == "__main__":
    build_gui()
