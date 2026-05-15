import numpy as np
import socket
from skimage.measure import label as sk_label

SERVER_IP = "84.237.21.36"
SERVER_PORT = 5152

def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def compute_center(img, labels, lbl):
    idx = np.unravel_index(np.argmax(img * (labels == lbl)), img.shape)
    return idx

def receive_bytes(sock, n):
    buffer = bytearray()
    while len(buffer) < n:
        chunk = sock.recv(n - len(buffer))
        if not chunk:
            return None
        buffer.extend(chunk)
    return buffer

def process_image(im) -> float:
    labeled_img = sk_label(im > 0)
    c1 = compute_center(im, labeled_img, 1)
    c2 = compute_center(im, labeled_img, 2)
    return euclidean_distance(c1, c2)

def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((SERVER_IP, SERVER_PORT))
        s.sendall(b"124ras1")
        print(s.recv(10))

        response = b"nope"
        while response != b"yep":
            s.sendall(b"get")
            data: None | bytearray = receive_bytes(s, 40002)
            if data is None:
                break

            rows, cols = data[0], data[1]
            img_array = np.frombuffer(data[2:], dtype=np.uint8).reshape(rows, cols)
            distance: float = round(process_image(img_array), 1)

            s.sendall(str(distance).encode())
            print(s.recv(10))
            s.sendall(b"beat")
            response = s.recv(10)

if __name__ == "__main__":
    main()
