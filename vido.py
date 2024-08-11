import os
import argparse
import cv2
import numpy as np
import importlib.util

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', required=True, help='Folder the .tflite file is located in')
parser.add_argument('--graph', default='detect.tflite', help='Name of the .tflite file')
parser.add_argument('--labels', default='labelmap.txt', help='Name of the labelmap file')
parser.add_argument('--threshold', type=float, default=0.5, help='Minimum confidence threshold for displaying detected objects')
parser.add_argument('--video', default='test.mp4', help='Name of the video file')
parser.add_argument('--edgetpu', action='store_true', help='Use Coral Edge TPU Accelerator to speed up detection')

args = parser.parse_args()

# Paths
MODEL_PATH = os.path.join(os.getcwd(), args.modeldir, args.graph)
LABEL_PATH = os.path.join(os.getcwd(), args.modeldir, args.labels)
VIDEO_PATH = os.path.join(os.getcwd(), args.video)

# Load label map
with open(LABEL_PATH, 'r') as f:
    labels = [line.strip() for line in f.readlines()]
    if labels[0] == '???':
        del labels[0]

# Load TensorFlow Lite model
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter, load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter, load_delegate

interpreter = Interpreter(model_path=MODEL_PATH, experimental_delegates=[load_delegate('libedgetpu.so.1.0')] if args.edgetpu else None)
interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height, width = input_details[0]['shape'][1], input_details[0]['shape'][2]
floating_model = input_details[0]['dtype'] == np.float32

# Map output details based on TensorFlow version
outname = output_details[0]['name']
boxes_idx, classes_idx, scores_idx = (1, 3, 0) if 'StatefulPartitionedCall' in outname else (0, 1, 2)

# Open video file
video = cv2.VideoCapture(VIDEO_PATH)
imW, imH = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        print('Reached the end of the video!')
        break

    frame_resized = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)
    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5

    # Perform detection
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Retrieve results
    boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]
    classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]
    scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]

    # Process detections
    for i in range(len(scores)):
        if scores[i] > args.threshold:
            ymin, xmin = int(max(1, (boxes[i][0] * imH))), int(max(1, (boxes[i][1] * imW)))
            ymax, xmax = int(min(imH, (boxes[i][2] * imH))), int(min(imW, (boxes[i][3] * imW)))
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 4)
            label = f"{labels[int(classes[i])]}: {int(scores[i] * 100)}%"
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (xmin, max(ymin, labelSize[1] + 10)), (xmin + labelSize[0], max(ymin, labelSize[1] + 10) + baseLine - 10), (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (xmin, max(ymin, labelSize[1] + 10) - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    cv2.imshow('Object detector', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# Cleanup
video.release()
cv2.destroyAllWindows()
