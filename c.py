from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FfmpegOutput
import time

def main():
    # Inisialisasi Picamera2
    picam2 = Picamera2()
    
    # Konfigurasi untuk merekam video
    video_config = picam2.create_video_configuration()
    picam2.configure(video_config)
    
    # Inisialisasi encoder H264
    encoder = H264Encoder(10000000)
    
    # Mulai kamera
    picam2.start()

    # Menyiapkan output video
    video_output = FfmpegOutput("video.mp4")
    
    # Mulai perekaman video
    print("Starting video recording.")
    picam2.start_recording(encoder, output=video_output)
    
    # Tunggu selama 5 detik
    time.sleep(5)
    
    # Hentikan perekaman
    picam2.stop_recording()
    print("Recording finished.")
    
    # Matikan kamera
    picam2.stop()

if __name__ == '__main__':
    main()
