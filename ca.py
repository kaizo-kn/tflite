from picamera2 import Picamera2
import time
import subprocess

# Inisialisasi Picamera2
picam2 = Picamera2()

# Konfigurasi untuk menangkap video
video_config = picam2.create_video_configuration()
picam2.configure(video_config)

# Mulai kamera
picam2.start()

# Mulai merekam video
picam2.start_recording("output.h264")

# Tunggu selama 3 detik
time.sleep(3)

# Hentikan perekaman
picam2.stop_recording()

# Matikan kamera
picam2.stop()

print("Video berhasil direkam dengan durasi 3 detik.")

# Konversi ke MP4 menggunakan ffmpeg
subprocess.run(['ffmpeg', '-i', 'output.h264', '-c:v', 'copy', 'output.mp4'])

print("Video berhasil dikonversi ke MP4.")
