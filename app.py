from flask import Flask, render_template, Response, jsonify, request, Blueprint
import numpy as np
import cv2
import dlib
import os
import time
import threading
import queue
import json
import uuid
import datetime
import traceback
import scipy.signal
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import gc  # Garbage collector

# Load environment variables
load_dotenv()

# Global variables for BPM sharing between threads
current_bpm = 0
bpm_ready = False
frame_queue = queue.Queue(maxsize=128)
signal_quality = 0
monitoring_active = False
bpm_history = []
session_id = None
face_detected = False
intensity_data = []
timestamps = []
last_calculation_time = 0
last_face_time = 0
app_start_time = time.time()

# Import configuration settings
from config import Config

# Initialize face detection
detector = None
predictor = None
opencv_face_cascade = None
face_detection_available = False

# Webcam Parameters
realWidth, realHeight = 640, 480
videoWidth, videoHeight = 320, 240  # Increased for better resolution

# Color Magnification Parameters
levels = 3
alpha = 170
minFrequency = 0.8  # Lower minimum for more sensitivity
maxFrequency = 2.5  # Higher maximum for wider range
bufferSize = 150
bufferIndex = 0

# Heart Rate Calculation Parameters
MAX_SAMPLES = 300
MIN_SAMPLES_FOR_CALCULATION = 30
MAX_TIME_WINDOW = 10.0

# Create a Blueprint for the BPM monitor
bpm_bp = Blueprint('bpm', __name__)

# Helper function to safely load the face detection models
def initialize_face_detection():
    """Initialize face detection with proper error handling and fallbacks"""
    global detector, predictor, opencv_face_cascade, face_detection_available
    
    print("Initializing face detection system...")
    
    # Initialize with sane defaults
    detector = None
    predictor = None
    opencv_face_cascade = None
    
    # Try to load dlib's face detector
    try:
        detector = dlib.get_frontal_face_detector()
        print("Dlib face detector loaded successfully")
        
        # Look for the shape predictor file
        shape_predictor_paths = [
            'shape_predictor_68_face_landmarks.dat',
            os.path.join(os.getcwd(), 'shape_predictor_68_face_landmarks.dat'),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'shape_predictor_68_face_landmarks.dat')
        ]
        
        for path in shape_predictor_paths:
            if os.path.exists(path):
                try:
                    predictor = dlib.shape_predictor(path)
                    print(f"Shape predictor loaded from: {path}")
                    break
                except Exception as e:
                    print(f"Failed to load shape predictor from {path}: {e}")
        
        if predictor is None:
            print("WARNING: shape_predictor_68_face_landmarks.dat not found.")
            print("Face detection will work, but landmark detection is disabled.")
    except Exception as e:
        print(f"Error initializing dlib face detector: {e}")
        traceback.print_exc()
        detector = None
    
    # Fall back to OpenCV's Haar Cascade if dlib fails
    if detector is None:
        try:
            print("Using OpenCV's Haar Cascade as fallback...")
            opencv_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            if not opencv_face_cascade.empty():
                print("Successfully loaded OpenCV face detector")
            else:
                print("ERROR: Failed to load OpenCV Haar Cascade")
                opencv_face_cascade = None
        except Exception as cascade_error:
            print(f"Failed to load OpenCV face detector: {cascade_error}")
            opencv_face_cascade = None
    
    face_detection_available = detector is not None or opencv_face_cascade is not None
    return face_detection_available

# Signal processing functions
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Apply a Butterworth bandpass filter to the data with improved error handling"""
    try:
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        
        # Apply reasonable limits to prevent instability
        low = max(0.001, min(low, 0.99))
        high = max(low + 0.001, min(high, 0.99))
        
        b, a = scipy.signal.butter(order, [low, high], btype='band')
        filtered_data = scipy.signal.filtfilt(b, a, data, padtype='even')
        return filtered_data
    except Exception as e:
        print(f"Error in bandpass filter: {e}")
        # Return original data if filtering fails
        return data

def buildGauss(frame, levels):
    """Build a Gaussian pyramid from an input frame"""
    pyramid = [frame]
    for _ in range(levels):
        frame = cv2.pyrDown(frame)
        pyramid.append(frame)
    return pyramid

def reconstructFrame(pyramid, index, levels):
    """Reconstruct a frame from a Gaussian pyramid level"""
    filteredFrame = pyramid[index]
    for _ in range(levels):
        filteredFrame = cv2.pyrUp(filteredFrame)
    try:
        return filteredFrame[:videoHeight, :videoWidth]
    except:
        # Handle dimension mismatch
        return cv2.resize(filteredFrame, (videoWidth, videoHeight))

def calculate_bpm(intensities, timestamps):
    """Calculate BPM from intensity data using improved signal processing"""
    global signal_quality
    
    if len(intensities) < MIN_SAMPLES_FOR_CALCULATION:
        return 0, 0
    
    try:
        # Calculate sampling rate from timestamps
        if len(timestamps) < 2:
            return 0, 0
            
        # Get time interval between samples
        time_range = timestamps[-1] - timestamps[0]
        if time_range <= 0:
            return 0, 0
            
        # Calculate sampling frequency
        fs = len(timestamps) / time_range
        
        # Normalize intensity values
        intensities = np.array(intensities)
        intensities = intensities - np.mean(intensities)
        
        # Apply bandpass filter (0.8-2.5 Hz corresponds to 48-150 BPM)
        filtered_data = butter_bandpass_filter(intensities, 0.8, 2.5, fs)
        
        # Apply Hamming window to reduce spectral leakage
        windowed_data = filtered_data * np.hamming(len(filtered_data))
        
        # Calculate FFT
        fft_data = np.abs(np.fft.rfft(windowed_data))
        
        # Get frequencies
        freqs = np.fft.rfftfreq(len(windowed_data), 1.0/fs)
        
        # Get the max frequency in the expected heart rate range (48-150 BPM)
        bpm_range = np.where((freqs >= 0.8) & (freqs <= 2.5))[0]
        if len(bpm_range) == 0:
            return 0, 0
            
        peak_idx = bpm_range[np.argmax(fft_data[bpm_range])]
        peak_freq = freqs[peak_idx]
        peak_bpm = peak_freq * 60
        
        # Calculate signal quality (normalized peak prominence)
        if len(bpm_range) > 1:
            peak_value = fft_data[peak_idx]
            mean_value = np.mean(fft_data[bpm_range])
            signal_quality = min(100, int((peak_value / mean_value) * 50))  # Scale appropriately
        else:
            signal_quality = 0
            
        return peak_bpm, signal_quality
    except Exception as e:
        print(f"Error calculating BPM: {e}")
        traceback.print_exc()
        return 0, 0

def detect_face(frame):
    """Detect and extract the face region from the frame with fallbacks"""
    global detector, predictor, opencv_face_cascade, last_face_time, face_detected
    
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_rect = None
        
        # Try dlib detector first
        if detector is not None:
            faces = detector(gray)
            if len(faces) > 0:
                face_rect = (faces[0].left(), faces[0].top(), 
                            faces[0].right(), faces[0].bottom())
                face_detected = True
                last_face_time = time.time()
        
        # Fall back to OpenCV if dlib fails
        if face_rect is None and opencv_face_cascade is not None:
            faces = opencv_face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
            if len(faces) > 0:
                x, y, w, h = faces[0]
                face_rect = (x, y, x+w, y+h)
                face_detected = True
                last_face_time = time.time()
        
        if face_rect is not None:
            x1, y1, x2, y2 = face_rect
            
            # Apply forehead region of interest (improved targeting)
            forehead_y1 = y1 + int((y2 - y1) * 0.1)  # Top 20% of face
            forehead_y2 = y1 + int((y2 - y1) * 0.3)
            forehead_roi = frame[forehead_y1:forehead_y2, x1:x2]
            
            # Mark forehead ROI on frame for visualization
            cv2.rectangle(frame, (x1, forehead_y1), (x2, forehead_y2), (0, 255, 0), 1)
            
            # Mark face on frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            
            # Extract ROI for BPM detection (if available)
            if forehead_roi.size > 0:
                return True, frame, forehead_roi, face_rect
                
        # No face detected or ROI extraction failed
        if time.time() - last_face_time > 3.0:
            face_detected = False
        
        return False, frame, None, None
    except Exception as e:
        print(f"Error in face detection: {e}")
        if time.time() - last_face_time > 3.0:
            face_detected = False
        return False, frame, None, None

def process_frames():
    """Process video frames to extract and analyze BPM data"""
    global frame_queue, current_bpm, bpm_ready, signal_quality
    global monitoring_active, intensity_data, timestamps, last_calculation_time
    global face_detected, bpm_history
    
    print("BPM processing thread started")
    
    # Force garbage collection time tracking
    last_gc_time = time.time()
    
    # Initialize variables for the Eulerian video magnification
    firstFrame = np.zeros((videoHeight, videoWidth, 3))
    firstGauss = buildGauss(firstFrame, levels + 1)[levels]
    videoGauss = np.zeros((bufferSize, firstGauss.shape[0], firstGauss.shape[1], 3))
    
    # Initialize bufferIndex
    bufferIndex = 0
    
    while True:
        try:
            # Perform garbage collection periodically
            if time.time() - last_gc_time > 30:
                gc.collect()
                last_gc_time = time.time()
            
            # Skip processing if monitoring is not active
            if not monitoring_active:
                # Clear data when not monitoring
                if len(intensity_data) > 0:
                    intensity_data = []
                    timestamps = []
                time.sleep(0.1)
                continue
            
            # Get a frame from the queue
            if frame_queue.empty():
                time.sleep(0.01)
                continue
                
            # Get the next frame
            frame = frame_queue.get()
            if frame is None:
                continue
            
            # Try to detect and extract the face region
            face_found, marked_frame, roi, face_rect = detect_face(frame)
            
            if face_found and roi is not None:
                # Resize ROI for consistent processing
                try:
                    detectionFrame = cv2.resize(roi, (videoWidth, videoHeight))
                except Exception as resize_error:
                    print(f"Error resizing ROI: {resize_error}")
                    continue
                
                # Extract the green channel and compute the mean
                green_channel = detectionFrame[:, :, 1]
                mean_intensity = np.mean(green_channel)
                
                # Store the intensity value and timestamp
                current_time = time.time()
                intensity_data.append(mean_intensity)
                timestamps.append(current_time)
                
                # Keep only the most recent data
                if len(intensity_data) > MAX_SAMPLES:
                    # Remove oldest data points
                    excess = len(intensity_data) - MAX_SAMPLES
                    intensity_data = intensity_data[excess:]
                    timestamps = timestamps[excess:]
                
                # Only calculate BPM periodically to avoid unnecessary processing
                if (current_time - last_calculation_time > 0.5 and 
                    len(intensity_data) >= MIN_SAMPLES_FOR_CALCULATION):
                    
                    bpm_value, quality = calculate_bpm(intensity_data, timestamps)
                    
                    if bpm_value > 0:
                        # Update the global BPM value
                        current_bpm = int(bpm_value)
                        signal_quality = quality
                        bpm_ready = True
                        
                        # Add to history (for charting)
                        bpm_history.append((datetime.datetime.now().timestamp(), current_bpm))
                        
                        # Keep history at a reasonable size
                        if len(bpm_history) > 60:  # 1 minute of history at 1 sample/sec
                            bpm_history = bpm_history[-60:]
                        
                        # Log for debugging
                        print(f"BPM: {current_bpm}, Quality: {signal_quality}%")
                    
                    last_calculation_time = current_time
                
                # Build Gaussian pyramid for this frame (for visualization)
                videoGauss[bufferIndex] = buildGauss(detectionFrame, levels + 1)[levels]
                
                # Increment buffer index
                bufferIndex = (bufferIndex + 1) % bufferSize
            else:
                # No face detected in this frame
                pass
                
        except Exception as e:
            print(f"Error in frame processing: {e}")
            traceback.print_exc()
            time.sleep(0.1)  # Short sleep on error

def gen_frames():
    """Generate frames for the video stream"""
    global monitoring_active, face_detected, current_bpm, signal_quality
    
    # Initialize webcam with auto-exposure disabled for better stability
    webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow on Windows
    webcam.set(cv2.CAP_PROP_FRAME_WIDTH, realWidth)
    webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, realHeight)
    webcam.set(cv2.CAP_PROP_FPS, 30)
    
    # Try to disable auto white-balance and auto exposure for better stability
    webcam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)  # 0 = manual exposure
    webcam.set(cv2.CAP_PROP_AUTO_WB, 0)  # 0 = manual white balance
    
    # Variables for FPS calculation
    fps = 0
    ptime = time.time()
    
    try:
        while True:
            success, frame = webcam.read()
            if not success:
                print("Failed to read frame from webcam")
                break
                
            # Calculate FPS
            ctime = time.time()
            fps = 1.0 / (ctime - ptime)
            ptime = ctime
            
            # Add status info to the frame
            display_frame = frame.copy()
            
            # Draw FPS
            cv2.putText(display_frame, f'FPS: {int(fps)}', (20, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display BPM when available
            if monitoring_active:
                if face_detected:
                    color = (0, 255, 0)  # Green when face detected
                    status = "Face Detected"
                else:
                    color = (0, 0, 255)  # Red when no face
                    status = "No Face Detected"
                
                cv2.putText(display_frame, status, (20, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                            
                # Display BPM and signal quality
                if current_bpm > 0:
                    # Color code based on signal quality
                    if signal_quality > 70:
                        quality_color = (0, 255, 0)  # Green for good signal
                    elif signal_quality > 40:
                        quality_color = (0, 255, 255)  # Yellow for medium signal
                    else:
                        quality_color = (0, 0, 255)  # Red for poor signal
                        
                    cv2.putText(display_frame, f'BPM: {current_bpm}', (20, 90), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, quality_color, 2)
                                
                    cv2.putText(display_frame, f'Signal: {signal_quality}%', (20, 120), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, quality_color, 2)
                else:
                    cv2.putText(display_frame, 'Calculating BPM...', (20, 90), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                cv2.putText(display_frame, 'Monitoring Inactive', (20, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # If monitoring is active, add frame to processing queue
            if monitoring_active and not frame_queue.full():
                if face_detection_available:
                    # Don't block if queue is full
                    try:
                        frame_queue.put(frame, block=False)
                    except queue.Full:
                        pass
            
            # Encode the frame to JPEG for streaming
            ret, buffer = cv2.imencode('.jpg', display_frame)
            if not ret:
                continue
                
            # Yield the frame for streaming
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                   
    except Exception as e:
        print(f"Error in frame generation: {e}")
        traceback.print_exc()
    finally:
        # Clean up
        webcam.release()

def save_session_data(session_id):
    """Save the BPM session data to a file"""
    if len(bpm_history) == 0:
        return
        
    try:
        session_data = {
            "session_id": session_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "bpm_data": [{"timestamp": ts, "bpm": bpm} for ts, bpm in bpm_history],
            "average_bpm": sum(bpm for _, bpm in bpm_history) / len(bpm_history),
            "min_bpm": min(bpm for _, bpm in bpm_history),
            "max_bpm": max(bpm for _, bpm in bpm_history)
        }
        
        # Create directory if it doesn't exist
        bpm_dir = os.path.join(Config.MEDICAL_DATA_PATH, 'bpm_sessions')
        os.makedirs(bpm_dir, exist_ok=True)
        
        # Save to file
        filename = os.path.join(bpm_dir, f"{int(time.time())}_{session_id}.json")
        with open(filename, 'w') as f:
            json.dump(session_data, f, indent=2)
            
        print(f"Session data saved to {filename}")
    except Exception as e:
        print(f"Error saving session data: {e}")

# Route handlers for the blueprint
@bpm_bp.route('/')
def index():
    """Serve the main page"""
    return render_template('combined-assessment.html')

@bpm_bp.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@bpm_bp.route('/get_bpm')
def get_bpm():
    """Return current BPM and signal quality data as JSON"""
    global current_bpm, bpm_ready, signal_quality, face_detected, monitoring_active, bpm_history
    
    # Create chart data from history
    chart_data = []
    if len(bpm_history) > 0:
        for ts, bpm in bpm_history:
            chart_data.append({"x": ts * 1000, "y": bpm})  # Convert to milliseconds for JS
    
    response = {
        "bpm": current_bpm if bpm_ready else 0,
        "signal_quality": signal_quality,
        "face_detected": face_detected,
        "monitoring_active": monitoring_active,
        "status": "normal" if 60 <= current_bpm <= 100 else "abnormal",
        "chart_data": chart_data
    }
    
    return jsonify(response)

@bpm_bp.route('/start_monitoring', methods=['POST'])
def start_monitoring():
    """Start the heart rate monitoring process"""
    global monitoring_active, bpm_ready, current_bpm, session_id
    
    # Generate a new session ID
    session_id = str(uuid.uuid4())
    
    monitoring_active = True
    bpm_ready = False
    current_bpm = 0
    
    # Clear history for new session
    bpm_history.clear()
    
    return jsonify({
        "status": "success", 
        "message": "Heart rate monitoring started",
        "session_id": session_id
    })

@bpm_bp.route('/stop_monitoring', methods=['POST'])
def stop_monitoring():
    """Stop the heart rate monitoring process"""
    global monitoring_active, session_id
    
    # Save the session data if needed
    if monitoring_active and session_id:
        save_session_data(session_id)
    
    monitoring_active = False
    return jsonify({"status": "success", "message": "Heart rate monitoring stopped"})

@bpm_bp.route('/health_check')
def health_check():
    """Health check endpoint for monitoring application status"""
    status = {
        "status": "healthy",
        "face_detection_available": face_detection_available,
        "monitoring_active": monitoring_active,
        "uptime": int(time.time() - app_start_time),
        "timestamp": datetime.datetime.now().isoformat()
    }
    return jsonify(status)

@bpm_bp.route('/analyze', methods=['POST'])
def analyze_symptoms():
    """Analyze symptoms from the symptom checker component"""
    try:
        # Get data from request
        data = request.get_json()
        symptoms = data.get('symptoms', [])
        
        # Check if we have symptoms
        if not symptoms:
            return jsonify({'error': 'No symptoms provided'}), 400
            
        # Return a sample response since we don't have the actual symptom analyzer
        # In a real app, this would connect to a proper medical analysis system
        return jsonify({
            'possible_conditions': [
                {'name': 'Common Cold', 'probability': 0.75},
                {'name': 'Seasonal Allergies', 'probability': 0.65},
                {'name': 'Influenza', 'probability': 0.45}
            ],
            'recommendations': [
                'Rest and stay hydrated',
                'Monitor your symptoms for any worsening',
                'Consult a healthcare provider if symptoms persist'
            ]
        })
    except Exception as e:
        print(f"Error analyzing symptoms: {e}")
        return jsonify({'error': str(e)}), 500

# Function to initialize and start the BPM processing thread
def start_bpm_monitor():
    """Initialize the BPM monitor and start the processing thread"""
    # Initialize face detection system
    initialize_face_detection()
    
    # Start BPM processing in a separate thread
    process_thread = threading.Thread(target=process_frames)
    process_thread.daemon = True
    process_thread.start()
    print("BPM monitor initialized and processing thread started")

# Function to add BPM monitoring to the Flask app
def create_bpm_monitor(app):
    """Register BPM monitoring capabilities with the Flask app"""
    # Register the blueprint with the app
    app.register_blueprint(bpm_bp, url_prefix='/')
    
    # Start the BPM monitor when app starts
    with app.app_context():
        start_bpm_monitor()
        
    return app

# For standalone testing - only used during development
if __name__ == '__main__':
    # Create a minimal Flask app for testing
    app = Flask(__name__)
    app.config.from_object(Config)
    Config.init_app(app)  # Initialize data directories
    
    # Register the blueprint directly
    app.register_blueprint(bpm_bp)
    
    # Initialize face detection
    initialize_face_detection()
    
    # Start the BPM processing thread
    process_thread = threading.Thread(target=process_frames)
    process_thread.daemon = True
    process_thread.start()
    
    # Start the Flask app
    print("Starting Flask application (development mode)...")
    app.run(debug=True, host='0.0.0.0', port=5000)