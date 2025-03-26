import cv2
import dlib
import numpy as np
import time
import logging
import threading
import queue
from scipy.signal import butter, filtfilt
import base64

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BPMMonitor:
    def __init__(self):
        """Initialize the BPM Monitor"""
        # Webcam Parameters
        self.width, self.height = 640, 480
        self.video_width, self.video_height = 320, 240
        self.video_channels = 3
        
        # Color Magnification Parameters
        self.levels = 3
        self.alpha = 170
        self.min_frequency = 0.83  # ~50 BPM
        self.max_frequency = 3.0   # ~180 BPM
        self.buffer_size = 150
        self.buffer_index = 0
        
        # BPM Calculation Variables
        self.fps = 15
        self.bpm_calculation_frequency = 10
        self.bpm_buffer_index = 0
        self.bpm_buffer_size = 10
        self.bpm_buffer = np.zeros((self.bpm_buffer_size))
        
        # Processing state
        self.is_processing = False
        self.current_bpm = 0
        self.bpm_ready = False
        self.frame_queue = queue.Queue(maxsize=10)
        self.processed_frame_queue = queue.Queue(maxsize=10)
        
        # Initialize face detector
        try:
            self.detector = dlib.get_frontal_face_detector()
            # Try to initialize predictor if shape predictor file exists
            try:
                self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
                self.use_landmarks = True
            except:
                self.use_landmarks = False
                logger.warning("Shape predictor file not found. Using basic face detection.")
        except Exception as e:
            logger.error(f"Error initializing face detector: {e}")
            raise
    
    def start_monitoring(self):
        """Start BPM monitoring in a separate thread"""
        if self.is_processing:
            return False
        
        # Clear queues
        while not self.frame_queue.empty():
            self.frame_queue.get()
        while not self.processed_frame_queue.empty():
            self.processed_frame_queue.get()
        
        # Reset BPM data
        self.bpm_buffer = np.zeros((self.bpm_buffer_size))
        self.bpm_buffer_index = 0
        self.buffer_index = 0
        self.bpm_ready = False
        
        # Start processing thread
        self.is_processing = True
        self.process_thread = threading.Thread(target=self._process_frames)
        self.process_thread.daemon = True
        self.process_thread.start()
        
        # Start webcam thread
        self.webcam_thread = threading.Thread(target=self._capture_frames)
        self.webcam_thread.daemon = True
        self.webcam_thread.start()
        
        return True
    
    def stop_monitoring(self):
        """Stop BPM monitoring"""
        self.is_processing = False
        return True
    
    def get_current_bpm(self):
        """Get the current BPM reading"""
        return {
            'bpm': int(self.current_bpm) if self.bpm_ready else 0,
            'ready': self.bpm_ready
        }
    
    def get_latest_frame(self):
        """Get the latest processed frame as base64 encoded image"""
        if not self.processed_frame_queue.empty():
            frame = self.processed_frame_queue.get()
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = base64.b64encode(buffer).decode('utf-8')
            return frame_bytes
        return None
    
    def _capture_frames(self):
        """Capture frames from webcam"""
        try:
            webcam = cv2.VideoCapture(0)
            webcam.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            webcam.set(cv2.CAP_PROP_FPS, self.fps)
            
            frame_time = 1.0 / self.fps  # Target time between frames
            
            while self.is_processing:
                # Maintain consistent frame rate
                start_time = time.time()
                
                ret, frame = webcam.read()
                if not ret:
                    logger.error("Failed to capture frame from webcam")
                    break
                
                # Make sure frame is the right type
                if frame.dtype != np.uint8:
                    frame = frame.astype(np.uint8)
                
                # Put frame in queue for processing
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
                
                # Sleep to maintain frame rate
                elapsed = time.time() - start_time
                if elapsed < frame_time:
                    time.sleep(frame_time - elapsed)
            
            webcam.release()
        except Exception as e:
            logger.error(f"Error capturing frames: {e}")
            self.is_processing = False
    
    def _process_frames(self):
        """Process frames to extract BPM"""
        try:
            # Initialize Gaussian pyramid
            first_frame = np.zeros((self.video_height, self.video_width, self.video_channels), dtype=np.uint8)
            first_gauss = self._build_gauss(first_frame, self.levels + 1)[self.levels]
            video_gauss = np.zeros((self.buffer_size, first_gauss.shape[0], first_gauss.shape[1], self.video_channels))
            
            # Bandpass filter frequencies
            frequencies = (self.fps * np.arange(self.buffer_size)) / self.buffer_size
            mask = (frequencies >= self.min_frequency) & (frequencies <= self.max_frequency)
            
            # Fourier transform averages
            fourier_transform_avg = np.zeros((self.buffer_size))
            
            # Processing variables
            i = 0
            ptime = time.time()
            
            while self.is_processing:
                if self.frame_queue.empty():
                    time.sleep(0.01)
                    continue
                
                # Get frame from queue
                frame = self.frame_queue.get()
                
                # Calculate FPS
                ctime = time.time()
                fps = 1 / (ctime - ptime)
                ptime = ctime
                
                # Make a copy for display
                display_frame = frame.copy()
                
                # Convert frame to grayscale for face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = self.detector(gray)
                
                # Draw FPS on frame
                cv2.putText(display_frame, f'FPS: {int(fps)}', (10, self.height - 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if faces:
                    # Process the first face detected
                    face = faces[0]
                    
                    # Get face region
                    x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
                    
                    # Draw rectangle around face
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    
                    # Extract and resize face region
                    detection_frame = frame[y1:y2, x1:x2]
                    if detection_frame.size == 0:
                        continue
                        
                    detection_frame = cv2.resize(detection_frame, (self.video_width, self.video_height))
                    
                    # Apply Eulerian Video Magnification
                    video_gauss[self.buffer_index] = self._build_gauss(detection_frame, self.levels + 1)[self.levels]
                    fourier_transform = np.fft.fft(video_gauss, axis=0)
                    
                    # Bandpass filter
                    fourier_transform[mask == False] = 0
                    
                    # Calculate heart rate
                    if self.buffer_index % self.bpm_calculation_frequency == 0:
                        i += 1
                        for buf in range(self.buffer_size):
                            fourier_transform_avg[buf] = np.real(fourier_transform[buf]).mean()
                        
                        # Find the frequency with the highest magnitude
                        hz = frequencies[np.argmax(fourier_transform_avg)]
                        bpm = 60.0 * hz
                        
                        # Apply some constraints
                        if 50 <= bpm <= 180:  # Typical human heart rate range
                            self.bpm_buffer[self.bpm_buffer_index] = bpm
                            self.bpm_buffer_index = (self.bpm_buffer_index + 1) % self.bpm_buffer_size
                    
                    # Amplify and reconstruct
                    filtered = np.real(np.fft.ifft(fourier_transform, axis=0)) * self.alpha
                    filtered_frame = self._reconstruct_frame(filtered, self.buffer_index, self.levels)
                    
                    output_frame = detection_frame + filtered_frame
                    output_frame = cv2.convertScaleAbs(output_frame)
                    
                    # Update buffer index
                    self.buffer_index = (self.buffer_index + 1) % self.buffer_size
                    
                    # Display processed face (resized)
                    output_frame_show = cv2.resize(output_frame, (self.video_width // 2, self.video_height // 2))
                    h, w = output_frame_show.shape[:2]
                    display_frame[10:10+h, self.width-10-w:self.width-10] = output_frame_show
                    
                    # Calculate and display BPM
                    if i > self.bpm_buffer_size:
                        # Calculate average BPM from buffer
                        nonzero_indices = self.bpm_buffer > 0
                        if np.any(nonzero_indices):
                            avg_bpm = self.bpm_buffer[nonzero_indices].mean()
                            self.current_bpm = avg_bpm
                            self.bpm_ready = True
                            
                            # Apply color based on BPM
                            if 60 <= avg_bpm <= 100:  # Normal range
                                bpm_color = (0, 255, 0)  # Green
                            elif avg_bpm < 60:  # Low
                                bpm_color = (255, 255, 0)  # Yellow
                            else:  # High
                                bpm_color = (0, 0, 255)  # Red
                                
                            cv2.putText(display_frame, f'BPM: {int(avg_bpm)}', (10, 50), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, bpm_color, 2)
                        else:
                            cv2.putText(display_frame, "Calculating BPM...", (10, 50), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    else:
                        cv2.putText(display_frame, "Calculating BPM...", (10, 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                else:
                    cv2.putText(display_frame, "No face detected", (10, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Put processed frame in queue for frontend
                if not self.processed_frame_queue.full():
                    self.processed_frame_queue.put(display_frame)
        
        except Exception as e:
            logger.error(f"Error processing frames: {e}")
            self.is_processing = False
    
    def _build_gauss(self, frame, levels):
        """Build a Gaussian pyramid for a frame"""
        pyramid = [frame]
        for i in range(levels):
            frame = cv2.pyrDown(frame)
            pyramid.append(frame)
        return pyramid
    
    def _reconstruct_frame(self, pyramid, index, levels):
        """Reconstruct frame from Gaussian pyramid"""
        filtered_frame = pyramid[index]
        for i in range(levels):
            filtered_frame = cv2.pyrUp(filtered_frame)
        
        # Ensure dimensions match before returning
        filtered_frame = filtered_frame[:self.video_height, :self.video_width]
        return filtered_frame 