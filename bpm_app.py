import time
import numpy as np
import cv2
import dlib
import sys
from flask import Flask, render_template, Response, jsonify
import threading
import queue

# Initialize Flask app
app = Flask(__name__)

# Global variables for BPM sharing between threads
current_bpm = 0
bpm_ready = False
frame_queue = queue.Queue(maxsize=10)

# Load Dlib's face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r'shape_predictor_68_face_landmarks.dat')

# Webcam Parameters
realWidth, realHeight = 640, 480
videoWidth, videoHeight = 160, 120
videoChannels, videoFrameRate = 3, 15

# Color Magnification Parameters
levels = 3
alpha = 170
minFrequency = 1.0
maxFrequency = 2.0
bufferSize = 150
bufferIndex = 0

# Helper Functions
def buildGauss(frame, levels):
    pyramid = [frame]
    for _ in range(levels):
        frame = cv2.pyrDown(frame)
        pyramid.append(frame)
    return pyramid

def reconstructFrame(pyramid, index, levels):
    filteredFrame = pyramid[index]
    for _ in range(levels):
        filteredFrame = cv2.pyrUp(filteredFrame)
    filteredFrame = filteredFrame[:videoHeight, :videoWidth]
    return filteredFrame

def process_frames():
    global current_bpm, bpm_ready
    
    # Gaussian Pyramid Initialization
    firstFrame = np.zeros((videoHeight, videoWidth, videoChannels), dtype=np.uint8)
    firstGauss = buildGauss(firstFrame, levels + 1)[levels]
    videoGauss = np.zeros((bufferSize, firstGauss.shape[0], firstGauss.shape[1], videoChannels))
    fourierTransformAvg = np.zeros((bufferSize))

    # Bandpass Filter for Specified Frequencies
    frequencies = (1.0 * videoFrameRate) * np.arange(bufferSize) / (1.0 * bufferSize)
    mask = (frequencies >= minFrequency) & (frequencies <= maxFrequency)

    # Heart Rate Calculation Variables
    bpmCalculationFrequency = 10
    bpmBufferIndex = 0
    bpmBufferSize = 10
    bpmBuffer = np.zeros((bpmBufferSize))

    i, bufferIndex = 0, 0
    
    # Initialize webcam
    webcam = cv2.VideoCapture(0)
    webcam.set(cv2.CAP_PROP_FRAME_WIDTH, realWidth)
    webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, realHeight)
    
    ptime = time.time()
    
    try:
        while True:
            # Capture frame
            ret, frame = webcam.read()
            if not ret:
                print("Failed to capture frame from webcam")
                break
                
            # Calculate FPS
            ftime = time.time()
            fps = 1 / (ftime - ptime)
            ptime = ftime
            
            # Make a copy for display
            display_frame = frame.copy()
            
            # Convert frame to correct format
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = detector(gray)
            
            # Draw FPS on frame
            cv2.putText(display_frame, f'FPS: {int(fps)}', (30, 440), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if faces:
                for face in faces:
                    # Get facial landmarks
                    landmarks = predictor(gray, face)
                    
                    # Extract face region
                    x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    
                    # Extract and resize the face region
                    detection_frame = frame[y1:y2, x1:x2]
                    if detection_frame.size == 0:  # Skip if face region is empty
                        continue
                        
                    detection_frame = cv2.resize(detection_frame, (videoWidth, videoHeight))
                    
                    # Apply the Eulerian Video Magnification
                    videoGauss[bufferIndex] = buildGauss(detection_frame, levels + 1)[levels]
                    fourierTransform = np.fft.fft(videoGauss, axis=0)
                    
                    # Bandpass filter
                    fourierTransform[mask == False] = 0
                    
                    # Calculate heart rate
                    if bufferIndex % bpmCalculationFrequency == 0:
                        i += 1
                        for buf in range(bufferSize):
                            fourierTransformAvg[buf] = np.real(fourierTransform[buf]).mean()
                        hz = frequencies[np.argmax(fourierTransformAvg)]
                        bpm = 60.0 * hz
                        bpmBuffer[bpmBufferIndex] = bpm
                        bpmBufferIndex = (bpmBufferIndex + 1) % bpmBufferSize
                    
                    # Amplify and reconstruct
                    filtered = np.real(np.fft.ifft(fourierTransform, axis=0)) * alpha
                    filtered_frame = reconstructFrame(filtered, bufferIndex, levels)
                    output_frame = detection_frame + filtered_frame
                    output_frame = cv2.convertScaleAbs(output_frame)
                    
                    # Update buffer index
                    bufferIndex = (bufferIndex + 1) % bufferSize
                    
                    # Display the processed face
                    output_frame_show = cv2.resize(output_frame, (videoWidth // 2, videoHeight // 2))
                    display_frame[0:videoHeight // 2, (realWidth - videoWidth // 2):realWidth] = output_frame_show
                    
                    # Calculate average BPM
                    bpm_value = bpmBuffer.mean()
                    
                    # Update the global BPM value
                    if i > bpmBufferSize:
                        current_bpm = int(bpm_value)
                        bpm_ready = True
                        cv2.putText(display_frame, f'BPM: {current_bpm}', 
                                    (videoWidth // 2, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                                    1, (0, 255, 0), 2)
                    else:
                        cv2.putText(display_frame, "Calculating BPM...", 
                                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                                    1, (0, 255, 255), 2)
            
            # Put the frame in the queue for web streaming
            if not frame_queue.full():
                frame_queue.put(display_frame)
            
    except Exception as e:
        print(f"Error in frame processing: {str(e)}")
    finally:
        webcam.release()

def generate_frames():
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            # If no frame available, send a blank frame
            blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank_frame, "Starting camera...", (180, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            _, buffer = cv2.imencode('.jpg', blank_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_bpm')
def get_bpm():
    if bpm_ready:
        return jsonify({'bpm': current_bpm, 'status': 'ready'})
    else:
        return jsonify({'bpm': 0, 'status': 'calculating'})

if __name__ == '__main__':
    # Start the frame processing in a separate thread
    processing_thread = threading.Thread(target=process_frames)
    processing_thread.daemon = True
    processing_thread.start()
    
    # Run the Flask app
    app.run(debug=False, threaded=True) 