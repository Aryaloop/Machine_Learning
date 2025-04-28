import os
import cv2
import numpy as np
from gtts import gTTS
import pickle
import face_recognition
import time

# Folder dataset foto wajah
KNOWN_FACES_DIR = "known_faces"
ENCODINGS_FILE = "encodings.pkl"

# Model YOLO untuk deteksi wajah
YOLO_CFG = "yolov3-face.cfg"
YOLO_WEIGHTS = "yolov3-wider_16000.weights"
YOLO_CLASSES = "coco.names"

def load_known_faces():
    """Memuat wajah-wajah yang dikenal dari file encoding."""
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, "rb") as f:
            known_faces, known_names = pickle.load(f)
            return known_faces, known_names
    else:
        known_faces = []
        known_names = []
        for filename in os.listdir(KNOWN_FACES_DIR):
            filepath = os.path.join(KNOWN_FACES_DIR, filename)
            image = face_recognition.load_image_file(filepath)
            encodings = face_recognition.face_encodings(image)
            if len(encodings) > 0:
                known_faces.append(encodings[0])
                known_names.append(os.path.splitext(filename)[0])
            else:
                print(f"Wajah tidak ditemukan di file {filename}.")
        with open(ENCODINGS_FILE, "wb") as f:
            pickle.dump((known_faces, known_names), f)
        return known_faces, known_names

def deteksi_wajah_yolo(frame, net, output_layers):
    """Deteksi wajah menggunakan YOLO."""
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(output_layers)

    boxes = []
    confidences = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]  # Skor kelas dimulai dari indeks ke-5
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:  # Threshold confidence
                box = detection[0:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Periksa apakah `indices` valid
    if len(indices) > 0 and isinstance(indices, np.ndarray):
        indices = indices.flatten()
        return [boxes[i] for i in indices]
    else:
        return []  # Tidak ada deteksi yang valid

def main():
    print("Sistem CCTV Rumah dengan Deteksi Wajah dan Peringatan")

    # Muat wajah-wajah yang dikenal
    known_faces, known_names = load_known_faces()

    # Load model YOLO
    net = cv2.dnn.readNetFromDarknet(YOLO_CFG, YOLO_WEIGHTS)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # Inisialisasi kamera
    video_capture = cv2.VideoCapture(0)  # Kamera default

    if not video_capture.isOpened():
        print("Kamera tidak tersedia!")
        return

    while True:
        # Baca frame dari kamera
        ret, frame = video_capture.read()
        if not ret:
            print("Gagal membaca frame dari kamera.")
            break

        # Deteksi wajah
        boxes = deteksi_wajah_yolo(frame, net, output_layers)
        for box in boxes:
            x, y, w, h = box
            face = frame[y:y + h, x:x + w]
            rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

            # Cocokkan dengan wajah yang dikenal
            encodings = face_recognition.face_encodings(rgb_face)
            if len(encodings) > 0:
                encoding = encodings[0]
                matches = face_recognition.compare_faces(known_faces, encoding, tolerance=0.6)
                name = "Tidak Dikenal"
                if True in matches:
                    match_index = matches.index(True)
                    name = known_names[match_index]
                else:
                    # Buat peringatan suara
                    tts = gTTS("Ada penyusup", lang="id")
                    tts.save("alert.mp3")
                    os.system("start alert.mp3")  # Windows
                    time.sleep(1)  # Jeda waktu 1 detik

                label = f"Dikenali: {name}" if name != "Tidak Dikenal" else "Wajah Tidak Dikenal"

                # Gambarkan kotak di sekitar wajah
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Tampilkan video real-time
        cv2.imshow("CCTV Rumah", frame)

        # Tekan 'q' untuk keluar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Lepaskan kamera dan tutup semua jendela
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
