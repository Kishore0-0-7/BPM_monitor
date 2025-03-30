import cv2
import dlib
import numpy as np
import time
import logging
import threading
import queue
from scipy.signal import butter, filtfilt, find_peaks
import base64
from datetime import datetime
import json
import os
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BPMMonitor:
    def __init__(self, data_dir=None):
        """Initialize the BPM Monitor
        
        Args:
            data_dir (str, optional): Directory to store BPM data. Defaults to None.
        """
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
        
        # Recording features
        self.recording_enabled = False
        self.bpm_history = []
        self.timestamp_history = []
        self.session_start_time = None
        
        # Data storage and reporting
        self.data_dir = data_dir if data_dir else os.path.join(os.getcwd(), 'data', 'bpm_sessions')
        os.makedirs(self.data_dir, exist_ok=True)
        self.session_id = None
        
        # Initialize face detector
        try:
            self.detector = dlib.get_frontal_face_detector()
            # Try to initialize predictor if shape predictor file exists
            shape_predictor_path = Path("shape_predictor_68_face_landmarks.dat")
            if shape_predictor_path.exists():
                self.predictor = dlib.shape_predictor(str(shape_predictor_path))
                self.use_landmarks = True
                logger.info("Using facial landmarks for enhanced BPM detection")
            else:
                self.use_landmarks = False
                logger.warning("Shape predictor file not found. Using basic face detection.")
        except Exception as e:
            logger.error(f"Error initializing face detector: {e}")
            raise
    
    def start_monitoring(self, record_data=False):
        """Start BPM monitoring in a separate thread
        
        Args:
            record_data (bool, optional): Whether to record BPM data for later analysis. Defaults to False.
        
        Returns:
            bool: Success status
        """
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
        
        # Setup recording if enabled
        self.recording_enabled = record_data
        if record_data:
            self.bpm_history = []
            self.timestamp_history = []
            self.session_start_time = datetime.now()
            self.session_id = self.session_start_time.strftime("%Y%m%d_%H%M%S")
            logger.info(f"Recording BPM data for session {self.session_id}")
        
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
        """Stop BPM monitoring
        
        Returns:
            bool: Success status
        """
        if not self.is_processing:
            return False
            
        self.is_processing = False
        
        # Save session data if recording was enabled
        if self.recording_enabled and self.session_id and len(self.bpm_history) > 0:
            self._save_session_data()
        
        return True
    
    def get_current_bpm(self):
        """Get the current BPM reading
        
        Returns:
            dict: BPM data and status
        """
        if self.bpm_ready:
            # Calculate heart rate variability
            hrv = self._calculate_hrv() if len(self.bpm_history) > 10 else 0
            
            return {
                'bpm': int(self.current_bpm),
                'ready': True,
                'zone': self._get_heart_rate_zone(self.current_bpm),
                'hrv': hrv
            }
        else:
            return {
                'bpm': 0,
                'ready': False,
                'zone': 'Unknown',
                'hrv': 0
            }
    
    def get_latest_frame(self):
        """Get the latest processed frame as base64 encoded image
        
        Returns:
            str: Base64 encoded JPEG image
        """
        if not self.processed_frame_queue.empty():
            frame = self.processed_frame_queue.get()
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = base64.b64encode(buffer).decode('utf-8')
            return frame_bytes
        return None
    
    def get_session_summary(self):
        """Get summary statistics for the current or last session
        
        Returns:
            dict: Summary statistics
        """
        if not self.bpm_history or len(self.bpm_history) == 0:
            return {
                'status': 'No data available',
                'avg_bpm': 0,
                'min_bpm': 0,
                'max_bpm': 0,
                'duration': 0,
                'samples': 0
            }
        
        # Calculate summary statistics
        avg_bpm = sum(self.bpm_history) / len(self.bpm_history)
        min_bpm = min(self.bpm_history)
        max_bpm = max(self.bpm_history)
        
        # Calculate duration
        if self.timestamp_history and len(self.timestamp_history) >= 2:
            duration = (self.timestamp_history[-1] - self.timestamp_history[0]).total_seconds()
        else:
            duration = 0
            
        return {
            'status': 'Recording' if self.is_processing and self.recording_enabled else 'Stopped',
            'session_id': self.session_id,
            'avg_bpm': round(avg_bpm, 1),
            'min_bpm': min_bpm,
            'max_bpm': max_bpm,
            'duration': round(duration, 1),
            'samples': len(self.bpm_history),
            'zones': self._calculate_zone_percentages()
        }
    
    def get_session_data(self, session_id=None):
        """Get detailed data for a specific session or the current session
        
        Args:
            session_id (str, optional): Session ID to retrieve. Defaults to None (current session).
            
        Returns:
            dict: Session data
        """
        # Return current session data if no ID provided and we have data
        if not session_id and self.bpm_history:
            return {
                'session_id': self.session_id,
                'start_time': self.session_start_time.isoformat() if self.session_start_time else None,
                'bpm_values': self.bpm_history,
                'timestamps': [ts.isoformat() for ts in self.timestamp_history],
                'summary': self.get_session_summary()
            }
        
        # Otherwise, load from file if it exists
        target_id = session_id or self.session_id
        if not target_id:
            return {'error': 'No session ID provided or available'}
            
        session_file = os.path.join(self.data_dir, f"{target_id}.json")
        if os.path.exists(session_file):
            try:
                with open(session_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading session data: {e}")
                return {'error': f"Could not load session data: {str(e)}"}
        else:
            return {'error': f"Session data not found for ID: {target_id}"}
    
    def list_sessions(self):
        """List all available BPM monitoring sessions
        
        Returns:
            list: List of session IDs and timestamps
        """
        sessions = []
        if os.path.exists(self.data_dir):
            for file in os.listdir(self.data_dir):
                if file.endswith('.json'):
                    session_id = file.replace('.json', '')
                    try:
                        with open(os.path.join(self.data_dir, file), 'r') as f:
                            data = json.load(f)
                            sessions.append({
                                'session_id': session_id,
                                'start_time': data.get('start_time', 'Unknown'),
                                'duration': data.get('summary', {}).get('duration', 0),
                                'avg_bpm': data.get('summary', {}).get('avg_bpm', 0)
                            })
                    except Exception as e:
                        logger.error(f"Error loading session data: {e}")
                        sessions.append({
                            'session_id': session_id,
                            'start_time': 'Unknown',
                            'error': str(e)
                        })
        
        # Sort by start time (most recent first)
        sessions.sort(key=lambda x: x.get('start_time', ''), reverse=True)
        return sessions
    
    def generate_report(self, session_id=None, output_format='json'):
        """Generate a report for a specific session
        
        Args:
            session_id (str, optional): Session ID. Defaults to None (current/last session).
            output_format (str, optional): Report format. Defaults to 'json'.
            
        Returns:
            dict or str: Report data or path to report file
        """
        # Get session data
        session_data = self.get_session_data(session_id)
        if 'error' in session_data:
            return session_data
        
        # Basic report in JSON format
        if output_format == 'json':
            return session_data
        
        # Generate report filename
        target_id = session_id or self.session_id
        if not target_id:
            return {'error': 'No session ID provided or available'}
            
        report_dir = os.path.join(self.data_dir, 'reports')
        os.makedirs(report_dir, exist_ok=True)
        
        # JSON file report
        if output_format == 'json_file':
            report_file = os.path.join(report_dir, f"{target_id}_report.json")
            try:
                with open(report_file, 'w') as f:
                    json.dump(session_data, f, indent=4)
                return {'report_file': report_file}
            except Exception as e:
                logger.error(f"Error generating report: {e}")
                return {'error': f"Could not generate report: {str(e)}"}
        
        # Unsupported format
        return {'error': f"Unsupported output format: {output_format}"}
    
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
                    
                    # Use facial landmarks if available
                    roi_frame = None
                    if self.use_landmarks:
                        try:
                            landmarks = self.predictor(gray, face)
                            # Get forehead region (above eyebrows)
                            forehead_y = landmarks.part(21).y  # eyebrow
                            forehead_h = max(30, int((forehead_y - face.top()) * 0.8))
                            forehead_x = landmarks.part(21).x
                            forehead_w = landmarks.part(22).x - forehead_x
                            
                            # Ensure we stay within face bounds
                            forehead_x = max(face.left(), forehead_x - forehead_w//2)
                            forehead_y = max(face.top(), forehead_y - forehead_h)
                            
                            # Get forehead ROI
                            roi_frame = frame[forehead_y:forehead_y+forehead_h, 
                                            forehead_x:forehead_x+forehead_w]
                            
                            # Draw ROI on display frame
                            cv2.rectangle(display_frame, 
                                        (forehead_x, forehead_y), 
                                        (forehead_x+forehead_w, forehead_y+forehead_h), 
                                        (0, 255, 0), 2)
                        except Exception as e:
                            logger.warning(f"Error using facial landmarks: {e}")
                            # Fall back to whole face
                            roi_frame = None
                    
                    if roi_frame is None or roi_frame.size == 0:
                        # Draw rectangle around face
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                        
                        # Extract and resize face region
                        roi_frame = frame[y1:y2, x1:x2]
                        if roi_frame.size == 0:
                            continue
                    
                    # Resize ROI to standard size
                    detection_frame = cv2.resize(roi_frame, (self.video_width, self.video_height))
                    
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
                            
                            # Record BPM if enabled
                            if self.recording_enabled:
                                self.bpm_history.append(int(avg_bpm))
                                self.timestamp_history.append(datetime.now())
                            
                            # Apply color based on BPM
                            hr_zone = self._get_heart_rate_zone(avg_bpm)
                            zone_colors = {
                                'Rest': (0, 255, 0),       # Green
                                'Light': (0, 255, 255),    # Yellow
                                'Moderate': (0, 165, 255), # Orange
                                'Vigorous': (0, 0, 255),   # Red
                                'Maximum': (0, 0, 128)     # Dark red
                            }
                            bpm_color = zone_colors.get(hr_zone, (0, 255, 0))
                                
                            cv2.putText(display_frame, f'BPM: {int(avg_bpm)}', (10, 50), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, bpm_color, 2)
                            cv2.putText(display_frame, f'Zone: {hr_zone}', (10, 90), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, bpm_color, 2)
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
    
    def _get_heart_rate_zone(self, bpm):
        """Determine heart rate zone based on BPM
        
        Args:
            bpm (float): Heart rate in beats per minute
            
        Returns:
            str: Heart rate zone
        """
        if bpm < 60:
            return 'Rest'
        elif bpm < 100:
            return 'Light'
        elif bpm < 140:
            return 'Moderate'
        elif bpm < 170:
            return 'Vigorous'
        else:
            return 'Maximum'
    
    def _calculate_hrv(self):
        """Calculate heart rate variability from BPM history
        
        Returns:
            float: Heart rate variability (RMSSD approximation)
        """
        if len(self.bpm_history) < 10:
            return 0
            
        # Convert BPM to approximate RR intervals (in ms)
        rr_intervals = [60000 / bpm for bpm in self.bpm_history[-30:]]
        
        # Calculate differences between successive RR intervals
        rr_diffs = np.diff(rr_intervals)
        
        # Root Mean Square of Successive Differences (RMSSD)
        rmssd = np.sqrt(np.mean(np.square(rr_diffs)))
        
        return round(rmssd, 2)
    
    def _calculate_zone_percentages(self):
        """Calculate percentage of time spent in each heart rate zone
        
        Returns:
            dict: Percentage of time in each zone
        """
        if not self.bpm_history:
            return {zone: 0 for zone in ['Rest', 'Light', 'Moderate', 'Vigorous', 'Maximum']}
            
        zone_counts = {
            'Rest': 0,
            'Light': 0,
            'Moderate': 0,
            'Vigorous': 0,
            'Maximum': 0
        }
        
        for bpm in self.bpm_history:
            zone = self._get_heart_rate_zone(bpm)
            zone_counts[zone] += 1
            
        total = len(self.bpm_history)
        zone_percentages = {zone: round(count * 100 / total, 1) for zone, count in zone_counts.items()}
        
        return zone_percentages
    
    def _save_session_data(self):
        """Save current session data to file"""
        if not self.session_id or len(self.bpm_history) == 0:
            return
            
        session_file = os.path.join(self.data_dir, f"{self.session_id}.json")
        
        try:
            # Create session data
            session_data = {
                'session_id': self.session_id,
                'start_time': self.session_start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'bpm_values': self.bpm_history,
                'timestamps': [ts.isoformat() for ts in self.timestamp_history],
                'summary': self.get_session_summary()
            }
            
            # Save to file
            with open(session_file, 'w') as f:
                json.dump(session_data, f, indent=4)
                
            logger.info(f"Saved BPM session data to {session_file}")
            
        except Exception as e:
            logger.error(f"Error saving session data: {e}") 