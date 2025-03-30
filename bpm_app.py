import time
import numpy as np
import cv2
import dlib
import sys
from flask import Flask, render_template, Response, jsonify
import threading
import queue
import os
import traceback
import glob
import scipy.signal
from scipy.fft import fft, fftfreq
import gc  # Garbage collector

# Initialize Flask app
app = Flask(__name__)

# Global variables for BPM sharing between threads
current_bpm = 0
bpm_ready = False
frame_queue = queue.Queue(maxsize=128)
signal_quality = 0
monitoring_active = False

# Enhanced error handling for face detector initialization
print("Initializing face detection system...")

# Initialize with sane defaults
detector = None
predictor = None

try:
    # Try to load dlib face detector first
    detector = dlib.get_frontal_face_detector()
    print("Face detector loaded successfully")
    
    # Try multiple potential paths for shape predictor
    shape_predictor_paths = [
        'shape_predictor_68_face_landmarks.dat',
        os.path.join(os.getcwd(), 'shape_predictor_68_face_landmarks.dat'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'shape_predictor_68_face_landmarks.dat')
    ]
    
    # Look for the file in common locations
    found_predictor = False
    for path in shape_predictor_paths:
        if os.path.exists(path):
            try:
                predictor = dlib.shape_predictor(path)
                print(f"Shape predictor loaded successfully from: {path}")
                found_predictor = True
                break
            except Exception as e:
                print(f"Failed to load shape predictor from {path}: {e}")
    
    if not found_predictor:
        # Try to find by pattern matching
        potential_files = glob.glob("**/shape_predictor*.dat", recursive=True)
        
        if potential_files:
            try:
                predictor = dlib.shape_predictor(potential_files[0])
                print(f"Shape predictor loaded from found file: {potential_files[0]}")
                found_predictor = True
            except Exception as e:
                print(f"Failed to load found shape predictor: {e}")
        
        if not found_predictor:
            print("WARNING: shape_predictor_68_face_landmarks.dat file not found.")
            print("Face detection will work, but landmark detection will be disabled.")
            print("Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
            predictor = None
            
            # Try to auto-download without prompting
            auto_download = os.environ.get('AUTO_DOWNLOAD_PREDICTOR', '').lower() == 'true'
            if auto_download:
                print("Automatically downloading shape_predictor_68_face_landmarks.dat...")
                try:
                    import urllib.request
                    import bz2
                    
                    url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
                    filename = "shape_predictor_68_face_landmarks.dat.bz2"
                    
                    # Download the file with a timeout
                    print("Downloading from dlib.net (this may take a while)...")
                    urllib.request.urlretrieve(url, filename)
                    
                    # Extract the file
                    print("Extracting the file...")
                    with bz2.BZ2File(filename, 'rb') as source, open("shape_predictor_68_face_landmarks.dat", "wb") as dest:
                        dest.write(source.read())
                    
                    # Try to load the extracted file
                    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
                    print("Successfully downloaded and loaded the predictor!")
                    found_predictor = True
                    
                    # Clean up the compressed file
                    os.remove(filename)
                except Exception as download_error:
                    print(f"Error during download: {download_error}")
                    print("Please download the file manually.")
            else:
                print("To auto-download the predictor file, set environment variable AUTO_DOWNLOAD_PREDICTOR=true")
except Exception as e:
    print(f"Error initializing face detector: {e}")
    traceback.print_exc()
    detector = None
    predictor = None
    
# Fall back to OpenCV's Haar Cascade if dlib fails
if detector is None:
    try:
        print("Attempting to use OpenCV's Haar Cascade as fallback...")
        opencv_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if opencv_face_cascade.empty():
            print("ERROR: Failed to load OpenCV Haar Cascade")
            # Try one more approach - direct path
            cascade_path = os.path.join(cv2.__path__[0], 'data', 'haarcascade_frontalface_default.xml')
            if os.path.exists(cascade_path):
                opencv_face_cascade = cv2.CascadeClassifier(cascade_path)
                if not opencv_face_cascade.empty():
                    print("Successfully loaded OpenCV face detector from explicit path")
        else:
            print("Successfully loaded OpenCV face detector as fallback")
    except Exception as cascade_error:
        print(f"Failed to initialize OpenCV face detector: {cascade_error}")
        print("WARNING: No face detection is available!")
        opencv_face_cascade = None

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
    global frame_queue, current_bpm, bpm_ready, signal_quality, monitoring_active
    
    # Variables for heart rate tracking
    forehead_intensities = []
    timestamps = []
    
    # Variables for face tracking
    last_face_rect = None
    last_face_time = 0
    lost_face_counter = 0
    
    # Heart rate calculation parameters
    MAX_SAMPLES = 300  # Maximum number of samples to store
    MIN_SAMPLES_FOR_CALCULATION = 30  # Reduced from 60 for faster response
    MAX_TIME_WINDOW = 10.0  # Reduced from 15.0 seconds to get faster results
    
    print("BPM processing thread started")
    
    # Periodically force garbage collection
    last_gc_time = time.time()
    
    # Time of last BPM calculation
    last_bpm_time = 0
    
    while True:
        try:
            # Perform garbage collection periodically
            if time.time() - last_gc_time > 30:
                gc.collect()
                last_gc_time = time.time()
            
            # Skip processing if monitoring is not active
            if not monitoring_active:
                time.sleep(0.1)
                continue
            
            # Get a frame from the queue
            if frame_queue.empty():
                time.sleep(0.01)
                continue
                
            frame, timestamp = frame_queue.get()
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Try to detect face
            face_detected = False
            face_rect = None
            
            # Try with dlib detector
            if detector is not None:
                try:
                    # Apply histogram equalization for better detection in different lighting
                    equalized = cv2.equalizeHist(gray)
                    
                    # Use more sensitive detection with higher upsampling (2 for better detection)
                    faces = detector(equalized, 2)
                    
                    if len(faces) > 0:
                        # Find largest face
                        largest_face = max(faces, key=lambda rect: (rect.right() - rect.left()) * (rect.bottom() - rect.top()))
                        face_rect = largest_face
                        face_detected = True
                        
                        # Get facial landmarks if available
                        landmarks = None
                        if predictor is not None:
                            try:
                                landmarks = predictor(gray, largest_face)
                            except Exception as e:
                                print(f"Landmark detection error: {e}")
                        
                        # Extract forehead region - either from landmarks or approximation
                        if landmarks is not None and landmarks.num_parts > 0:
                            # Using landmarks for more precise forehead extraction
                            try:
                                # Get points for the eyebrows and top of nose
                                points = []
                                for i in range(17, 27):  # Eyebrow and top nose points
                                    points.append((landmarks.part(i).x, landmarks.part(i).y))
                                
                                if points:
                                    # Calculate forehead region
                                    min_x = min(p[0] for p in points)
                                    max_x = max(p[0] for p in points)
                                    min_y = min(p[1] for p in points) - int((largest_face.bottom() - largest_face.top()) * 0.2)
                                    min_y = max(0, min_y)  # Ensure we don't go outside frame
                                    max_y = min(p[1] for p in points)
                                    
                                    # Extract forehead region
                                    forehead_roi = frame[min_y:max_y, min_x:max_x]
                                else:
                                    # Fall back to using approximation
                                    raise ValueError("No landmark points found")
                            except Exception as landmark_error:
                                print(f"Error using landmarks: {landmark_error}")
                                # Fall back to approximation method
                                forehead_height = int((largest_face.bottom() - largest_face.top()) * 0.25)
                                forehead_top = max(0, largest_face.top())
                                forehead_bottom = min(forehead_top + forehead_height, frame.shape[0])
                                
                                # Use middle 60% width to avoid including hair/background
                                face_width = largest_face.right() - largest_face.left()
                                forehead_width = int(face_width * 0.6)
                                forehead_left = max(0, largest_face.left() + int(face_width * 0.2))
                                forehead_right = min(forehead_left + forehead_width, frame.shape[1])
                                
                                # Extract forehead region
                                forehead_roi = frame[forehead_top:forehead_bottom, forehead_left:forehead_right]
                        else:
                            # Approximate forehead region (top 1/4 of face)
                            forehead_height = int((largest_face.bottom() - largest_face.top()) * 0.25)
                            forehead_top = max(0, largest_face.top())
                            forehead_bottom = min(forehead_top + forehead_height, frame.shape[0])
                            
                            # Use middle 60% width to avoid including hair/background
                            face_width = largest_face.right() - largest_face.left()
                            forehead_width = int(face_width * 0.6)
                            forehead_left = max(0, largest_face.left() + int(face_width * 0.2))
                            forehead_right = min(forehead_left + forehead_width, frame.shape[1])
                            
                            # Extract forehead region
                            forehead_roi = frame[forehead_top:forehead_bottom, forehead_left:forehead_right]
                        
                        # Calculate average green channel intensity in forehead region
                        if forehead_roi.size > 0:  # Ensure region is not empty
                            # Use the green channel which is most sensitive to blood flow changes
                            green_intensity = np.mean(forehead_roi[:, :, 1])  # Green channel
                            
                            # Basic sanity check to filter out extreme outliers
                            if len(forehead_intensities) > 0:
                                last_intensity = forehead_intensities[-1]
                                # If the change is too extreme, it's likely an error
                                if abs(green_intensity - last_intensity) > last_intensity * 0.5:
                                    # Skip this frame if it's an outlier
                                    continue
                            
                            # Store intensity and timestamp
                            forehead_intensities.append(green_intensity)
                            timestamps.append(timestamp)
                            
                            # Keep arrays at a reasonable size
                            if len(forehead_intensities) > MAX_SAMPLES:
                                # Remove oldest samples
                                forehead_intensities = forehead_intensities[-MAX_SAMPLES:]
                                timestamps = timestamps[-MAX_SAMPLES:]
                                
                            # Also trim samples older than MAX_TIME_WINDOW
                            if len(timestamps) > 2 and timestamps[-1] - timestamps[0] > MAX_TIME_WINDOW:
                                # Find index of first sample within window
                                cutoff_time = timestamps[-1] - MAX_TIME_WINDOW
                                idx = 0
                                while idx < len(timestamps) and timestamps[idx] < cutoff_time:
                                    idx += 1
                                
                                # Trim arrays
                                forehead_intensities = forehead_intensities[idx:]
                                timestamps = timestamps[idx:]
                        
                        # Reset lost face counter
                        lost_face_counter = 0
                        last_face_time = timestamp
                except Exception as e:
                    print(f"Dlib detection error: {e}")
            
            # If no face detected with dlib, try OpenCV
            if not face_detected and 'opencv_face_cascade' in globals() and opencv_face_cascade is not None:
                try:
                    # Try with OpenCV's face detector as fallback with multiple scale parameters
                    opencv_faces = opencv_face_cascade.detectMultiScale(
                        gray, 
                        scaleFactor=1.1, 
                        minNeighbors=5,
                        minSize=(30, 30)
                    )
                    
                    # If no faces detected, try with more aggressive parameters
                    if len(opencv_faces) == 0:
                        opencv_faces = opencv_face_cascade.detectMultiScale(
                            cv2.equalizeHist(gray),  # Apply histogram equalization 
                            scaleFactor=1.05,        # Smaller scale factor (more sensitive)
                            minNeighbors=3,          # Fewer neighbors required
                            minSize=(20, 20)         # Allow smaller faces
                        )
                    
                    if len(opencv_faces) > 0:
                        # Find largest face
                        largest_face = max(opencv_faces, key=lambda rect: rect[2] * rect[3])
                        x, y, w, h = largest_face
                        
                        # Create forehead region (top 1/4 of face)
                        forehead_height = int(h * 0.25)
                        forehead_top = max(0, y)
                        forehead_bottom = min(y + forehead_height, frame.shape[0])
                        
                        # Use middle 60% width to avoid hair/background
                        forehead_width = int(w * 0.6)
                        forehead_left = max(0, x + int(w * 0.2))
                        forehead_right = min(forehead_left + forehead_width, frame.shape[1])
                        
                        # Extract forehead region
                        forehead_roi = frame[forehead_top:forehead_bottom, forehead_left:forehead_right]
                        
                        if forehead_roi.size > 0:
                            # Calculate average green channel intensity
                            green_intensity = np.mean(forehead_roi[:, :, 1])
                            
                            # Basic sanity check to filter out extreme outliers
                            if len(forehead_intensities) > 0:
                                last_intensity = forehead_intensities[-1]
                                # If the change is too extreme, it's likely an error
                                if abs(green_intensity - last_intensity) > last_intensity * 0.5:
                                    # Skip this frame if it's an outlier
                                    continue
                            
                            # Store intensity and timestamp
                            forehead_intensities.append(green_intensity)
                            timestamps.append(timestamp)
                            
                            # Keep arrays at reasonable size
                            if len(forehead_intensities) > MAX_SAMPLES:
                                forehead_intensities = forehead_intensities[-MAX_SAMPLES:]
                                timestamps = timestamps[-MAX_SAMPLES:]
                        
                        # Reset lost face counter
                        lost_face_counter = 0
                        last_face_time = timestamp
                        face_detected = True
                except Exception as e:
                    print(f"OpenCV detection error: {e}")
            
            # If face not detected for too long, clear samples
            if not face_detected:
                lost_face_counter += 1
                if lost_face_counter > 30:  # About 1 second at 30fps
                    forehead_intensities = []
                    timestamps = []
                    current_bpm = 0
                    bpm_ready = False
                    signal_quality = 0
            
            # Calculate BPM periodically or if we have enough new samples
            current_time = time.time()
            enough_new_samples = len(forehead_intensities) >= MIN_SAMPLES_FOR_CALCULATION and len(forehead_intensities) % 10 == 0
            enough_time_passed = current_time - last_bpm_time > 0.5  # Recalculate every 0.5 seconds max
            
            if enough_new_samples and enough_time_passed and len(forehead_intensities) >= MIN_SAMPLES_FOR_CALCULATION:
                bpm, quality = calculate_bpm(forehead_intensities, timestamps)
                last_bpm_time = current_time
                
                # Update global variables with new BPM if valid
                if bpm > 0:
                    # Apply smoothing for stability
                    if current_bpm > 0:
                        # More weight to new value if quality is high, less if low
                        alpha = 0.3 + (quality / 200.0)  # Range 0.3-0.8
                        current_bpm = alpha * bpm + (1 - alpha) * current_bpm
                    else:
                        current_bpm = bpm
                    
                    signal_quality = quality
                    bpm_ready = True
                    print(f"BPM calculated: {current_bpm}, quality: {quality}%")
            
            # Short sleep to prevent high CPU usage
            time.sleep(0.01)
            
        except Exception as e:
            print(f"Error in process_frames: {e}")
            traceback.print_exc()
            time.sleep(0.1)  # Sleep longer on error to avoid rapid error loops

def calculate_bpm(intensities, timestamps):
    """
    Calculate BPM from intensity data using FFT and bandpass filtering
    
    Args:
        intensities: List of intensity values over time
        timestamps: List of timestamps corresponding to intensities
    
    Returns:
        Tuple of (bpm_value, signal_quality)
    """
    try:
        # Need at least 2 seconds of data for reasonable BPM calculation
        if len(intensities) < 15 or len(timestamps) < 15:  # Reduced minimum samples
            return 0, 0
        
        # Convert lists to numpy arrays for processing
        intensities = np.array(intensities)
        timestamps = np.array(timestamps)
        
        # Make sure we have enough time elapsed
        time_span = timestamps[-1] - timestamps[0]
        if time_span < 2.0:  # Reduced from 3.0 seconds to 2.0 seconds
            return 0, 0
        
        # Calculate sampling rate (fps) from timestamps
        time_diff = np.diff(timestamps)
        if len(time_diff) == 0:
            return 0, 0
        
        fps = 1.0 / np.mean(time_diff)
        
        # Detrend the signal to remove slow drifts and lighting changes
        detrended = scipy.signal.detrend(intensities)
        
        # Normalize intensity values
        normalized = detrended - np.mean(detrended)
        if np.std(normalized) != 0:
            normalized = normalized / np.std(normalized)
        
        # Apply bandpass filter to isolate heart rate frequencies (0.6-4Hz or 36-240 BPM)
        # Wider frequency range to capture more potential heart rates
        filtered_data = butter_bandpass_filter(normalized, 0.6, 4.0, fps, order=3)
        
        # Compute FFT
        n = len(filtered_data)
        fft_data = fft(filtered_data)
        freq = fftfreq(n, d=1.0/fps)
        
        # Only consider positive frequencies in physiological range (0.6-4Hz or 36-240 BPM)
        positive_freq_mask = (freq > 0.6) & (freq < 4.0)
        freq_positive = freq[positive_freq_mask]
        fft_positive = np.abs(fft_data[positive_freq_mask])
        
        # If no valid frequencies found
        if len(freq_positive) == 0:
            return 0, 0
        
        # Find peaks in the FFT data
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(fft_positive, 
                                      height=0.2*np.max(fft_positive),  # Lower threshold
                                      distance=3,  # Allow peaks to be closer
                                      prominence=0.1)  # Lower prominence threshold
        
        # Calculate signal quality metric (0-100%)
        if len(peaks) > 0 and np.max(fft_positive) > 0:
            # Signal quality based on peak prominence and SNR
            peak_idx = np.argmax(fft_positive)
            peak_value = fft_positive[peak_idx]
            noise_level = np.mean(fft_positive)
            
            if noise_level > 0:
                snr = peak_value / noise_level
                signal_quality = min(100, int(snr * 30))  # Scale SNR to 0-100%
            else:
                signal_quality = 0
        else:
            signal_quality = 0
        
        # If peaks found, get the dominant frequency
        if len(peaks) > 0:
            # Find the peak with maximum amplitude
            max_peak_idx = np.argmax(fft_positive[peaks])
            dominant_peak_idx = peaks[max_peak_idx]
            dominant_freq = freq_positive[dominant_peak_idx]
            bpm = 60.0 * dominant_freq  # Convert Hz to BPM
            
            # Verify the BPM is in a reasonable range (36-220 BPM) - wider range
            if 36 <= bpm <= 220:
                return round(bpm, 1), max(10, signal_quality)  # Minimum quality of 10%
        
        # Apply alternative method if FFT doesn't give good results
        # Use time-domain peak detection as fallback
        peaks, _ = find_peaks(filtered_data, distance=int(fps/4.5))  # Reduced minimum distance
        
        if len(peaks) >= 2:
            # Calculate average time between peaks
            peak_intervals = np.diff(timestamps[peaks])
            if len(peak_intervals) > 0 and np.mean(peak_intervals) > 0:
                mean_interval = np.mean(peak_intervals)
                bpm = 60.0 / mean_interval  # Convert seconds to BPM
                
                # Verify the BPM is in a reasonable range
                if 36 <= bpm <= 220:  # Wider range
                    return round(bpm, 1), max(10, signal_quality)  # Minimum 10% quality for this method
        
        # If all methods fail
        return 0, 0
        
    except Exception as e:
        print(f"Error in calculate_bpm: {e}")
        traceback.print_exc()
        return 0, 0

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Apply a Butterworth bandpass filter to the data with improved error handling
    
    Args:
        data: Input signal data
        lowcut: Lower frequency bound (Hz)
        highcut: Upper frequency bound (Hz)
        fs: Sampling frequency (Hz)
        order: Filter order
        
    Returns:
        Filtered signal data
    """
    try:
        from scipy.signal import butter, filtfilt
        
        # Handle edge cases
        if len(data) < 2 * order + 1:
            # Not enough data points for the filter, return original
            return data
            
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        
        # Prevent filter instability by constraining frequencies
        low = max(0.001, min(low, 0.98))
        high = max(low + 0.05, min(high, 0.98))
        
        # Create filter with error handling
        try:
            b, a = butter(order, [low, high], btype='band')
        except Exception as e:
            print(f"Error creating filter: {e}, using lower order")
            # Try with lower order if standard order fails
            b, a = butter(max(1, order-2), [low, high], btype='band')
        
        # Apply the filter
        padlen = min(len(data)-1, 3*order)  # Choose appropriate padding
        filtered = filtfilt(b, a, data, padlen=padlen)
        
        # Check for NaN values in the output (can happen with unstable filters)
        if np.any(np.isnan(filtered)):
            print("Warning: NaN values detected in filter output, returning original signal")
            return data
            
        return filtered
    except Exception as e:
        print(f"Error in bandpass filter: {e}")
        # Return original data on error
        return data

def gen_frames():
    """Generator function that yields encoded video frames for streaming"""
    global current_bpm, bpm_ready, frame_queue, monitoring_active, signal_quality
    
    # Initialize camera
    camera = None
    try:
        # Try different camera indices if initial one fails
        for cam_index in [0, 1, 2]:
            camera = cv2.VideoCapture(cam_index)
            if camera.isOpened():
                print(f"Camera initialized with index {cam_index}")
                # Set camera properties
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, realWidth)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, realHeight)
                camera.set(cv2.CAP_PROP_FPS, videoFrameRate)
                break
            else:
                print(f"Failed to open camera with index {cam_index}")
        
        if not camera.isOpened():
            print("Error: Could not open any camera")
            # Return a static error frame
            error_img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_img, "Camera Error", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            _, error_frame = cv2.imencode('.jpg', error_img)
            yield (b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + error_frame.tobytes() + b'\r\n')
            return
    except Exception as e:
        print(f"Camera initialization error: {e}")
        error_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_img, "Camera Error", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        _, error_frame = cv2.imencode('.jpg', error_img)
        yield (b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + error_frame.tobytes() + b'\r\n')
        return
    
    # Initialize variables for heart rate detection
    firstFrame = np.zeros((videoHeight, videoWidth, videoChannels))
    firstGauss = buildGauss(firstFrame, levels + 1)[levels]
    videoGauss = np.zeros((bufferSize, firstGauss.shape[0], firstGauss.shape[1], videoChannels))
    fourierTransformAvg = np.zeros((bufferSize))
    
    # Variables for BPM calculation
    bpmCalculationFrequency = 10
    bpmBufferIndex = 0
    bpmBufferSize = 10
    bpmBuffer = np.zeros((bpmBufferSize))
    
    i = 0
    last_face_detection_time = 0
    forehead_points = []
    last_update_time = time.time()
    fps_history = []
    
    try:
        while True:
            # Measure FPS
            current_time = time.time()
            time_diff = current_time - last_update_time
            if time_diff > 0:
                fps = 1.0 / time_diff
                fps_history.append(fps)
                if len(fps_history) > 30:  # Keep only last 30 FPS readings
                    fps_history.pop(0)
            last_update_time = current_time
            
            # Get a frame from the video stream
            success, frame = camera.read()
            if not success:
                print("Failed to grab frame")
                break
            
            # Resize frame to reduce processing load
            frame_small = cv2.resize(frame, (videoWidth, videoHeight))
            
            # Add current frame to processing queue if monitoring is active
            if monitoring_active:
                try:
                    # Add frame to queue without blocking
                    if not frame_queue.full():
                        frame_queue.put((frame_small.copy(), time.time()), block=False)
                except Exception as e:
                    print(f"Queue error: {e}")
            
            # Perform visualization on the main frame
            # Draw BPM value on frame if available
            if bpm_ready and current_bpm > 0:
                # Determine heart rate category for color-coding
                if current_bpm < 60:
                    color = (255, 0, 0)  # Blue - Low heart rate
                    status = "Low"
                elif current_bpm < 100:
                    color = (0, 255, 0)  # Green - Normal heart rate
                    status = "Normal"
                elif current_bpm < 140:
                    color = (0, 165, 255)  # Orange - Elevated heart rate
                    status = "Elevated"
                else:
                    color = (0, 0, 255)  # Red - High heart rate
                    status = "High"
                
                # Draw BPM value and status
                cv2.putText(frame, f"BPM: {int(current_bpm)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(frame, status, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
                # Draw signal quality indicator
                quality_text = f"Signal: {signal_quality}%"
                cv2.putText(frame, quality_text, (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Visualize monitoring status
                if monitoring_active:
                    status_text = "Monitoring: Active"
                    status_color = (0, 255, 0)  # Green
                else:
                    status_text = "Monitoring: Inactive"
                    status_color = (0, 0, 255)  # Red
                cv2.putText(frame, status_text, (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            
            else:
                # If BPM not yet calculated, show status message
                if monitoring_active:
                    cv2.putText(frame, "Calculating BPM...", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                else:
                    cv2.putText(frame, "Press 'Start Monitoring'", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Calculate and display FPS
            if fps_history:
                avg_fps = sum(fps_history) / len(fps_history)
                cv2.putText(frame, f"FPS: {int(avg_fps)}", (frame.shape[1] - 150, frame.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Encode frame to JPEG for streaming
            try:
                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    print("Failed to encode frame")
                    continue
                    
                # Yield the encoded frame for streaming
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            except Exception as e:
                print(f"Frame encoding error: {e}")
                continue
                
    except Exception as e:
        print(f"Stream error: {e}")
        traceback.print_exc()
    finally:
        # Clean up
        if camera is not None and camera.isOpened():
            camera.release()
        print("Camera released")

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(gen_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_bpm', methods=['GET'])
def get_bpm():
    """Return current BPM and signal quality data as JSON"""
    global current_bpm, signal_quality, bpm_ready, monitoring_active
    
    # Calculate heart rate category
    category = "unknown"
    if current_bpm > 0:
        if current_bpm < 60:
            category = "low"
        elif current_bpm < 100:
            category = "normal"
        elif current_bpm < 140:
            category = "elevated"
        else:
            category = "high"
    
    # Format BPM value for display
    bpm_display = int(current_bpm) if current_bpm > 0 else "--"
    
    # Return JSON response
    return jsonify({
        'bpm': current_bpm,
        'bpm_display': bpm_display,
        'signal_quality': signal_quality,
        'ready': bpm_ready,
        'monitoring': monitoring_active,
        'category': category,
        'timestamp': time.time()
    })

@app.route('/start_monitoring', methods=['POST'])
def start_monitoring():
    """Start the heart rate monitoring process"""
    global monitoring_active, bpm_ready, current_bpm, signal_quality
    
    monitoring_active = True
    
    # Reset BPM values when starting new monitoring session
    bpm_ready = False
    current_bpm = 0
    signal_quality = 0
    
    # Store the start time for UI status messages
    app.start_time = time.time()
    
    return jsonify({
        'status': 'success',
        'message': 'Heart rate monitoring started',
        'timestamp': app.start_time
    })

@app.route('/stop_monitoring', methods=['POST'])
def stop_monitoring():
    """Stop the heart rate monitoring process"""
    global monitoring_active
    monitoring_active = False
    return jsonify({'status': 'success', 'message': 'Heart rate monitoring stopped'})

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring application status"""
    # Check if the camera is available
    camera_status = "unknown"
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            camera_status = "available"
            success, _ = cap.read()
            if not success:
                camera_status = "error_reading"
        else:
            camera_status = "unavailable"
        cap.release()
    except Exception as e:
        camera_status = f"error: {str(e)}"
    
    # Check if face detection is working
    face_detection_status = "unknown"
    if detector is not None:
        face_detection_status = "available"
    elif 'opencv_face_cascade' in globals() and opencv_face_cascade is not None:
        face_detection_status = "fallback_available"
    else:
        face_detection_status = "unavailable"
    
    # Check landmark predictor
    landmark_status = "unknown"
    if predictor is not None:
        landmark_status = "available"
    else:
        landmark_status = "unavailable"
    
    # Memory usage info
    memory_info = {}
    try:
        import psutil
        process = psutil.Process()
        memory_info = {
            'rss': process.memory_info().rss / 1024 / 1024,  # MB
            'vms': process.memory_info().vms / 1024 / 1024,  # MB
            'percent': process.memory_percent()
        }
    except ImportError:
        memory_info = {'error': 'psutil not available'}
    
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'camera': camera_status,
        'face_detection': face_detection_status,
        'landmark_detection': landmark_status,
        'memory': memory_info,
        'monitoring_active': monitoring_active,
        'bpm_ready': bpm_ready
    })

@app.route('/reset', methods=['POST'])
def reset_application():
    """Reset the application state and clear any cached data"""
    global monitoring_active, current_bpm, bpm_ready, signal_quality, frame_queue
    
    # Stop monitoring
    monitoring_active = False
    
    # Reset BPM values
    current_bpm = 0
    bpm_ready = False
    signal_quality = 0
    
    # Clear frame queue
    try:
        while not frame_queue.empty():
            frame_queue.get_nowait()
    except Exception as e:
        print(f"Error clearing queue: {e}")
    
    # Force garbage collection
    gc.collect()
    
    return jsonify({
        'status': 'success',
        'message': 'Application reset complete',
        'timestamp': time.time()
    })

if __name__ == '__main__':
    try:
        # Start the BPM processing thread
        process_thread = threading.Thread(target=process_frames)
        process_thread.daemon = True
        process_thread.start()
        
        # Test thread for signal generation (for debugging)
        def test_thread_func():
            global current_bpm, bpm_ready, signal_quality
            while True:
                if not monitoring_active:
                    time.sleep(1)
                    continue
                # For debugging: Generate fake BPM data if enabled
                if os.environ.get('BPM_DEBUG_MODE') == 'true':
                    current_bpm = 60 + 20 * np.sin(time.time() / 10.0)
                    signal_quality = 70 + 20 * np.sin(time.time() / 5.0)
                    bpm_ready = True
                time.sleep(1)
        
        # Start processing wrapper
        def processing_wrapper():
            try:
                process_frames()
            except Exception as e:
                print(f"Processing thread crashed: {e}")
                traceback.print_exc()
        
        process_thread = threading.Thread(target=processing_wrapper)
        process_thread.daemon = True
        process_thread.start()
        
        # Optional test thread
        if os.environ.get('BPM_DEBUG_MODE') == 'true':
            test_thread = threading.Thread(target=test_thread_func)
            test_thread.daemon = True
            test_thread.start()
        
        # Handle graceful shutdown
        import signal
        def signal_handler(sig, frame):
            print("Shutting down...")
            # Clean up code here
            sys.exit(0)
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Initialize app start time
        app.start_time = time.time()
        
        # Start Flask app
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except Exception as e:
        print(f"Error starting application: {e}")
        traceback.print_exc()
        print("Application cannot continue and will exit.")
        sys.exit(1) 