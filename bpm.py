import time
import numpy as np
import cv2
import dlib
import sys
import scipy.signal as signal
from collections import deque

# Load Dlib's face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r'shape_predictor_68_face_landmarks.dat')

# Webcam Parameters
realWidth, realHeight = 640, 480
videoWidth, videoHeight = 160, 120
videoChannels, videoFrameRate = 3, 15

webcam = cv2.VideoCapture(0)
webcam.set(3, realWidth)
webcam.set(4, realHeight)

# Color Magnification Parameters
levels = 3
alpha = 170
minFrequency = 0.8  # Changed from 1.0 to detect slower heartbeats
maxFrequency = 2.0
bufferSize = 250  # Increased buffer size for better frequency resolution
bufferIndex = 0

# BPM Calculation and Display Parameters
bpmCalculationFrequency = 15
bpmBufferSize = 20  # Increased buffer size for more stable results
bpmBuffer = deque(maxlen=bpmBufferSize)  # Using deque for easier management
bpm_history = deque(maxlen=30)  # Store recent BPM values for display smoothing
display_bpm = 0  # Smoothed BPM for display

# Kalman filter for BPM tracking
# State: [BPM, BPM_change_rate]
kalman = cv2.KalmanFilter(2, 1)
kalman.measurementMatrix = np.array([[1.0, 0.0]], np.float32)
kalman.transitionMatrix = np.array([[1.0, 1.0], [0.0, 1.0]], np.float32)
kalman.processNoiseCov = np.array([[1e-4, 0.0], [0.0, 1e-3]], np.float32)
kalman.measurementNoiseCov = np.array([[0.1]], np.float32)

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
    return filteredFrame[:videoHeight, :videoWidth]

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.filtfilt(b, a, data, axis=0)
    return y

def process_bpm(value):
    """Process BPM using Kalman filter and exponential smoothing"""
    global display_bpm
    
    # Ignore extreme values
    if value < 40 or value > 180:
        return display_bpm
        
    # Update Kalman filter
    kalman.correct(np.array([[value]], np.float32))
    prediction = kalman.predict()
    filtered_bpm = prediction[0, 0]
    
    # Constrain to reasonable range
    filtered_bpm = max(50, min(filtered_bpm, 150))
    
    # Add to history
    bpm_history.append(filtered_bpm)
    
    # Calculate smoothed value using exponential smoothing
    alpha = 0.3  # Smoothing factor (lower = more smoothing)
    if display_bpm == 0:  # Initialize if first value
        display_bpm = filtered_bpm
    else:
        display_bpm = alpha * filtered_bpm + (1 - alpha) * display_bpm
        
    # Ensure value stays in physiological range
    display_bpm = max(50, min(display_bpm, 150))
    
    return round(display_bpm, 1)

# Gaussian Pyramid Initialization
firstFrame = np.zeros((videoHeight, videoWidth, videoChannels))
firstGauss = buildGauss(firstFrame, levels + 1)[levels]
videoGauss = np.zeros((bufferSize, firstGauss.shape[0], firstGauss.shape[1], videoChannels))
fourierTransformAvg = np.zeros((bufferSize))

# Bandpass Filter for Specified Frequencies
frequencies = (1.0 * videoFrameRate) * np.arange(bufferSize) / (1.0 * bufferSize)
mask = (frequencies >= minFrequency) & (frequencies <= maxFrequency)

i, ptime, ftime = 0, 0, 0
last_bpm_update_time = 0
face_detected = False

# Create a window with track bars for adjustment
cv2.namedWindow("Heart Rate Monitor")
cv2.createTrackbar("Min Freq", "Heart Rate Monitor", int(minFrequency * 10), 30, lambda x: None)
cv2.createTrackbar("Max Freq", "Heart Rate Monitor", int(maxFrequency * 10), 40, lambda x: None)
cv2.createTrackbar("Amplification", "Heart Rate Monitor", alpha, 300, lambda x: None)

print("Starting heart rate monitoring...")
print("Press 'q' to quit")

while True:
    ret, frame = webcam.read()
    if not ret:
        break
    
    # Get trackbar values
    minFrequency = cv2.getTrackbarPos("Min Freq", "Heart Rate Monitor") / 10.0
    maxFrequency = cv2.getTrackbarPos("Max Freq", "Heart Rate Monitor") / 10.0
    alpha = cv2.getTrackbarPos("Amplification", "Heart Rate Monitor")
    
    # Update mask based on trackbar values
    mask = (frequencies >= minFrequency) & (frequencies <= maxFrequency)
    
    # Calculate FPS
    ftime = time.time()
    fps = 1 / (ftime - ptime) if (ftime - ptime) > 0 else 30
    ptime = ftime
    
    # Overlay FPS
    cv2.putText(frame, f'FPS: {int(fps)}', (30, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Face Detection
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if gray.dtype != np.uint8:
            gray = gray.astype(np.uint8)
            
        faces = detector(gray)
        face_detected = len(faces) > 0
    except Exception as e:
        print(f"Error in face detection: {e}")
        face_detected = False
    
    if face_detected:
        for face in faces:
            try:
                landmarks = predictor(gray, face)
                
                # Extract face region
                x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
                
                # Draw rectangle around face
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                
                # Use forehead region (between eyebrows and hairline)
                forehead_y1 = landmarks.part(21).y - int((landmarks.part(21).y - face.top()) * 0.5)
                forehead_y2 = landmarks.part(21).y
                forehead_x1 = landmarks.part(21).x
                forehead_x2 = landmarks.part(22).x
                
                # Ensure coordinates are valid
                forehead_y1 = max(face.top(), forehead_y1)
                forehead_x1 = max(face.left(), forehead_x1)
                forehead_x2 = min(face.right(), forehead_x2)
                
                # Draw forehead region
                cv2.rectangle(frame, (forehead_x1, forehead_y1), (forehead_x2, forehead_y2), (0, 255, 0), 1)
                
                # Extract and resize forehead region
                detectionFrame = frame[forehead_y1:forehead_y2, forehead_x1:forehead_x2]
                if detectionFrame.size == 0:
                    # Fallback to full face if forehead region is invalid
                    detectionFrame = frame[y1:y2, x1:x2]
                
                # Make sure detectionFrame is not empty
                if detectionFrame.size > 0:
                    detectionFrame = cv2.resize(detectionFrame, (videoWidth, videoHeight))
                    
                    # Extract green channel (most relevant for heart rate)
                    green_frame = detectionFrame[:,:,1]
                    
                    # Apply additional preprocessing to green channel
                    green_frame = cv2.GaussianBlur(green_frame, (5, 5), 0)
                    
                    # Construct Gaussian Pyramid
                    videoGauss[bufferIndex] = buildGauss(detectionFrame, levels + 1)[levels]
                    
                    # Apply temporal filtering
                    if i >= bufferSize:
                        # Apply bandpass filter to temporal signal
                        filtered_video = butter_bandpass_filter(
                            videoGauss, minFrequency, maxFrequency, videoFrameRate)
                        videoGauss = filtered_video
                    
                    # Apply FFT
                    fourierTransform = np.fft.fft(videoGauss, axis=0)
                    
                    # Bandpass Filtering
                    fourierTransform[mask == False] = 0
                    
                    # Calculate Heart Rate periodically
                    current_time = time.time()
                    if (current_time - last_bpm_update_time) >= 1.0 / bpmCalculationFrequency:
                        last_bpm_update_time = current_time
                        i += 1
                        
                        # Process FFT data
                        for buf in range(bufferSize):
                            fourierTransformAvg[buf] = np.real(fourierTransform[buf]).mean()
                        
                        # Apply additional smoothing to frequency domain
                        fourierTransformAvg = signal.savgol_filter(fourierTransformAvg, 15, 3)
                        
                        # Find dominant frequency
                        max_idx = np.argmax(fourierTransformAvg)
                        
                        # Verify it's not just noise by checking prominence
                        if fourierTransformAvg[max_idx] > 1.5 * np.median(fourierTransformAvg):
                            hz = frequencies[max_idx]
                            bpm = 60.0 * hz
                            
                            # Only accept physiologically plausible values
                            if 50 <= bpm <= 150:
                                bpmBuffer.append(bpm)
                                
                    # Only calculate if we have enough data
                    if len(bpmBuffer) >= bpmBufferSize // 2:
                        # Sort and take median of middle values (trimmed mean)
                        sorted_bpms = sorted(bpmBuffer)
                        trim_amount = len(sorted_bpms) // 4
                        trimmed_bpms = sorted_bpms[trim_amount:-trim_amount] if trim_amount > 0 else sorted_bpms
                        bpm_value = sum(trimmed_bpms) / len(trimmed_bpms)
                        
                        # Process the BPM value for display
                        display_bpm = process_bpm(bpm_value)
                    
                    # Amplify and Reconstruct
                    filtered = np.real(np.fft.ifft(fourierTransform, axis=0)) * alpha
                    filteredFrame = reconstructFrame(filtered, bufferIndex, levels)
                    outputFrame = detectionFrame + filteredFrame
                    outputFrame = cv2.convertScaleAbs(outputFrame)
                    
                    # Display the processed frame in corner
                    bufferIndex = (bufferIndex + 1) % bufferSize
                    outputFrame_show = cv2.resize(outputFrame, (videoWidth // 2, videoHeight // 2))
                    frame[0:videoHeight // 2, (realWidth - videoWidth // 2):realWidth] = outputFrame_show
                    
                    # Display BPM
                    if i > bpmBufferSize and display_bpm > 0:
                        cv2.putText(frame, f'BPM: {display_bpm}', (30, 70), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
                        
                        # Add status based on BPM
                        if display_bpm < 60:
                            status = "Low Heart Rate"
                            color = (0, 255, 255)  # Yellow
                        elif display_bpm < 100:
                            status = "Normal Heart Rate"
                            color = (0, 255, 0)  # Green
                        elif display_bpm < 120:
                            status = "Elevated Heart Rate"
                            color = (0, 165, 255)  # Orange
                        else:
                            status = "High Heart Rate"
                            color = (0, 0, 255)  # Red
                            
                        cv2.putText(frame, status, (30, 120), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    else:
                        cv2.putText(frame, "Calculating BPM...", (30, 70), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            except Exception as e:
                print(f"Error processing face: {e}")
    else:
        # No face detected
        cv2.putText(frame, "No Face Detected", (30, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Display instructions
    cv2.putText(frame, "Adjust sliders for better results", (30, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    
    # Display the frame
    cv2.imshow("Heart Rate Monitor", frame)
    
    # Check for quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
webcam.release()
cv2.destroyAllWindows()