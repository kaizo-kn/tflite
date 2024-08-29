from picamera2 import Picamera2
import time

def main():
    # Inisialisasi Picamera2
    picam2 = Picamera2()
    
    # Konfigurasi untuk menangkap gambar
    still_config = picam2.create_still_configuration()
    picam2.configure(still_config)
    
    # Mulai kamera
    picam2.start()

    # Tunggu sebentar untuk kamera stabil
    time.sleep(2)

    # Tangkap gambar
    output_path = "cam-image.jpg"
    print(f"Capturing image to {output_path}.")
    picam2.capture_file(output_path)
    
    # Matikan kamera
    picam2.stop()
    print("Image capture finished.")

if __name__ == '__main__':
    main()
