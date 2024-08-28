import cv2
import numpy as np
import os

# Muat model TensorFlow
# model = tf.saved_model.load('path/to/saved_model')

# Fungsi untuk memproses dan mendeteksi objek
def detect_objects(frame):
    # Resize frame sesuai input model
    input_frame = cv2.resize(frame, (224, 224))  # Sesuaikan dengan input model
    input_frame = np.expand_dims(input_frame, axis=0)  # Tambahkan dimensi batch
    input_frame = input_frame / 255.0  # Normalisasi

    # Lakukan prediksi dengan model
    # predictions = model(input_frame)
    
    # Cetak hasil prediksi (ini harus disesuaikan dengan output model)
    print("Prediksi: 0")

    # Tambahkan kode untuk menggambar bounding box atau hasil lain pada frame

    return frame

# Menggunakan VideoCapture untuk menangkap aliran video
cap = cv2.VideoCapture('libcamera-vid -t 0 --inline --nopreview -o - | ffmpeg -i - -f rawvideo -pix_fmt bgr24 -')

if not cap.isOpened():
    print("Tidak dapat membuka kamera")
    exit()

# Loop untuk menangkap frame secara terus-menerus
while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal menangkap frame")
        break

    # Deteksi objek pada frame
    # processed_frame = detect_objects(frame)

    # Tampilkan frame yang telah diproses
    # cv2.imshow('Deteksi Objek Real-time', processed_frame)

    # Tekan 'q' untuk keluar dari loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan resource
cap.release()
cv2.destroyAllWindows()
