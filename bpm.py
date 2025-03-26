import time
import numpy as np
import cv2
import dlib
import sys

# Load Dlib's face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r'D:\CodeSpace\BPM\shape_predictor_68_face_landmarks.dat')


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
    return filteredFrame[:videoHeight, :videoWidth]

# Gaussian Pyramid Initialization
firstFrame = np.zeros((videoHeight, videoWidth, videoChannels))
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

i, ptime, ftime = 0, 0, 0

while True:
    ret, frame = webcam.read()
    if not ret:
        break
    
    # Debug: Print frame info
    print(f"Frame shape: {frame.shape}, dtype: {frame.dtype}")
    
    # Make sure frame is in the right format
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Ensure gray is the right type
    if gray.dtype != np.uint8:
        gray = gray.astype(np.uint8)
        
    # Convert to dlib format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Try both gray and rgb to see which works
    try:
        faces = detector(gray)
    except RuntimeError:
        print("Trying RGB format instead...")
        faces = detector(rgb_frame)
    
    ftime = time.time()
    fps = 1 / (ftime - ptime)
    ptime = ftime

    cv2.putText(frame, f'FPS: {int(fps)}', (30, 440), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if faces:
        for face in faces:
            landmarks = predictor(gray, face)

            # Extract face region based on landmarks (e.g., bounding box)
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)

            detectionFrame = frame[y1:y2, x1:x2]
            detectionFrame = cv2.resize(detectionFrame, (videoWidth, videoHeight))

            # Construct Gaussian Pyramid
            videoGauss[bufferIndex] = buildGauss(detectionFrame, levels + 1)[levels]
            fourierTransform = np.fft.fft(videoGauss, axis=0)

            # Bandpass Filtering
            fourierTransform[mask == False] = 0

            # Calculate Heart Rate
            if bufferIndex % bpmCalculationFrequency == 0:
                i += 1
                for buf in range(bufferSize):
                    fourierTransformAvg[buf] = np.real(fourierTransform[buf]).mean()
                hz = frequencies[np.argmax(fourierTransformAvg)]
                bpm = 60.0 * hz
                bpmBuffer[bpmBufferIndex] = bpm
                bpmBufferIndex = (bpmBufferIndex + 1) % bpmBufferSize

            # Amplify and Reconstruct
            filtered = np.real(np.fft.ifft(fourierTransform, axis=0)) * alpha
            filteredFrame = reconstructFrame(filtered, bufferIndex, levels)
            outputFrame = detectionFrame + filteredFrame
            outputFrame = cv2.convertScaleAbs(outputFrame)

            bufferIndex = (bufferIndex + 1) % bufferSize
            outputFrame_show = cv2.resize(outputFrame, (videoWidth // 2, videoHeight // 2))
            frame[0:videoHeight // 2, (realWidth - videoWidth // 2):realWidth] = outputFrame_show

            bpm_value = bpmBuffer.mean()

            if i > bpmBufferSize:
                cv2.putText(frame, f'BPM: {int(bpm_value)}', (videoWidth // 2, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Calculating BPM...", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Heart Rate Monitor", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()