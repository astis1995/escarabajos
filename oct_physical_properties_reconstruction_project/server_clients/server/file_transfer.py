import os
import zipfile

def zip_folder(folder_path, zip_name):
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, os.path.dirname(folder_path))
                zipf.write(full_path, rel_path)

def prepare_file(path):
    if os.path.isdir(path):
        zip_name = os.path.basename(path.rstrip('/\\')) + '.zip'
        zip_folder(path, zip_name)
        return zip_name
    else:
        return path

def send_file(conn, filepath, log):
    filename = os.path.basename(filepath)
    filesize = os.path.getsize(filepath)

    conn.send(len(filename).to_bytes(4, 'big'))
    conn.send(filename.encode())
    conn.send(filesize.to_bytes(8, 'big'))

    with open(filepath, 'rb') as f:
        while True:
            data = f.read(4096)
            if not data:
                break
            conn.sendall(data)
    log(f"✅ Enviado: {filename}")

def receive_file(conn, dest_folder, log):
    os.makedirs(dest_folder, exist_ok=True)

    filename_len = int.from_bytes(conn.recv(4), 'big')
    filename = conn.recv(filename_len).decode()
    filesize = int.from_bytes(conn.recv(8), 'big')

    filepath = os.path.join(dest_folder, filename)

    with open(filepath, 'wb') as f:
        received = 0
        while received < filesize:
            data = conn.recv(min(4096, filesize - received))
            if not data:
                break
            f.write(data)
            received += len(data)

    if filename.endswith('.zip'):
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(dest_folder)
        log(f"📦 ZIP extraído en {dest_folder}")
    log(f"✅ Recibido: {filename}")
