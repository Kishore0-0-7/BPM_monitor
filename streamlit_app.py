import streamlit as st
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime, timedelta
import cv2
import base64
import tempfile
from pathlib import Path
from fpdf import FPDF
import sys
import io
from PIL import Image
import uuid

# Don't try to import from app modules, use the mock implementations directly
class BPMMonitor:
    def __init__(self, data_dir=None):
        self.data_dir = data_dir or os.path.join(os.getcwd(), 'data', 'bpm_sessions')
        os.makedirs(self.data_dir, exist_ok=True)
        self.is_processing = False
        self.current_bpm = 70
        self.bpm_ready = True
        self.session_id = None
        self.session_start_time = None
        
        # Camera and processing variables
        self.cap = None
        self.frame_buffer = []
        self.buffer_size = 30  # Number of frames to keep for processing
        self.last_frame = None
        
        # ROI for face detection
        self.face_roi = None
        self.forehead_roi = None
        
        # Signal processing
        self.signal_buffer = []
        self.timestamps = []
        self.bpm_history = []
        self.hrv_data = []
        
    def start_monitoring(self, record_data=True, camera_index=0):
        self.is_processing = True
        self.session_start_time = datetime.now()
        self.session_id = self.session_start_time.strftime("%Y%m%d_%H%M%S")
        
        # Initialize the webcam
        try:
            self.cap = cv2.VideoCapture(camera_index)  # Use the selected camera
            # Set resolution for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Clear buffers
            self.frame_buffer = []
            self.signal_buffer = []
            self.timestamps = []
            self.bpm_history = []
            self.hrv_data = []
        except Exception as e:
            print(f"Error initializing webcam: {e}")
            # If real camera fails, still allow app to work with mock data
            self.cap = None
            
        return True
        
    def stop_monitoring(self):
        self.is_processing = False
        # Release the camera
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        return True
    
    def _extract_forehead_roi(self, frame):
        """Extract forehead region of interest (ROI) for pulse detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Improve contrast with histogram equalization
        gray = cv2.equalizeHist(gray)
        
        # Initialize forehead ROI
        forehead_roi = None
        
        # For tracking face detection stability
        if not hasattr(self, 'last_face_rect'):
            self.last_face_rect = None
            self.face_detect_counter = 0
            
        # Try to use haar cascade face detection (faster than dlib)
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Adjust detection parameters for better results
            min_neighbors = 5
            scale_factor = 1.1
            
            # Detect faces with updated parameters
            faces = face_cascade.detectMultiScale(gray, scale_factor, min_neighbors, 
                                               minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
            
            if len(faces) > 0:
                # Get the largest face
                (x, y, w, h) = max(faces, key=lambda rect: rect[2] * rect[3])
                
                # Add a margin around the face
                margin_x = int(w * 0.05)
                margin_y = int(h * 0.05)
                x = max(0, x - margin_x)
                y = max(0, y - margin_y)
                w = min(frame.shape[1] - x, w + 2 * margin_x)
                h = min(frame.shape[0] - y, h + 2 * margin_y)
                
                # Save the face rectangle for use in future frames
                self.last_face_rect = (x, y, w, h)
                self.face_detect_counter = min(self.face_detect_counter + 1, 10)
                
                # Define forehead as top 1/3 of face with improved precision
                forehead_height = int(h * 0.3)  # Use top 30% for better forehead targeting
                forehead_y = y
                
                # Adjust forehead region to be in the upper part of the face
                forehead_roi = (x, forehead_y, w, forehead_height)
                
                # Draw rectangles around face and forehead
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.rectangle(frame, (x, forehead_y), (x+w, forehead_y+forehead_height), (255, 0, 0), 2)
                
                # Add a label
                cv2.putText(frame, "Face Detected", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                return frame, forehead_roi
            elif self.last_face_rect is not None and self.face_detect_counter > 0:
                # If we lost tracking, use the last known position for a few frames
                self.face_detect_counter -= 1
                x, y, w, h = self.last_face_rect
                
                # Define forehead as before
                forehead_height = int(h * 0.3)
                forehead_roi = (x, y, w, forehead_height)
                
                # Draw rectangles with orange color to show it's using the cached position
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 165, 255), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+forehead_height), (0, 165, 255), 2)
                
                # Add a label
                cv2.putText(frame, "Tracking Face", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
                
                return frame, forehead_roi
                
        except Exception as e:
            print(f"Error in face detection: {e}")
        
        # Try using dlib as backup if available
        try:
            import dlib
            if hasattr(self, 'detector') and self.detector is not None:
                # Use dlib's face detector if it's available
                dlib_faces = self.detector(gray)
                if len(dlib_faces) > 0:
                    face = dlib_faces[0]  # Use the first face
                    x, y = face.left(), face.top()
                    w, h = face.width(), face.height()
                    
                    # Save for future tracking
                    self.last_face_rect = (x, y, w, h)
                    self.face_detect_counter = min(self.face_detect_counter + 1, 10)
                    
                    # Define forehead
                    forehead_height = int(h * 0.3)
                    forehead_roi = (x, y, w, forehead_height)
                    
                    # Draw rectangles
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 2)
                    cv2.rectangle(frame, (x, y), (x+w, y+forehead_height), (255, 0, 255), 2)
                    
                    # Add a label
                    cv2.putText(frame, "Dlib Face", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
                    
                    return frame, forehead_roi
        except ImportError:
            # Dlib not available, continue with default ROI
            pass
        
        # If we couldn't detect a face, use default ROI (center of frame) with text indicator
        h, w = frame.shape[:2]
        default_roi = (w//4, h//4, w//2, h//4)
        
        # Draw the default ROI rectangle
        x, y, roi_w, roi_h = default_roi
        cv2.rectangle(frame, (x, y), (x+roi_w, y+roi_h), (0, 0, 255), 2)
        cv2.putText(frame, "No face detected - using default ROI", (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                   
        return frame, default_roi
    
    def _process_roi(self, frame, roi):
        """Extract color signal from ROI"""
        if roi is None:
            return 0
            
        x, y, w, h = roi
        roi_data = frame[y:y+h, x:x+w]
        
        # Extract average green channel value (most sensitive to blood flow)
        green_val = np.mean(roi_data[:, :, 1])  # Green is index 1 in BGR
        return green_val
    
    def _calculate_bpm(self):
        """Calculate BPM from the green channel signal"""
        if len(self.signal_buffer) < 10:
            return self.current_bpm
            
        # Get time intervals
        time_range = self.timestamps[-1] - self.timestamps[0]
        if time_range < 1.0:  # Need at least 1 second of data
            return self.current_bpm
            
        # Normalize signal
        signal = np.array(self.signal_buffer)
        
        # Apply detrending to remove slow drifts
        detrended_signal = signal - np.polyval(np.polyfit(np.arange(len(signal)), signal, 2), np.arange(len(signal)))
        
        # Normalize to zero mean and unit variance
        normalized_signal = (detrended_signal - np.mean(detrended_signal)) / np.std(detrended_signal)
        
        # Apply bandpass filtering to isolate heart rate frequencies (0.7-4 Hz or 42-240 BPM)
        try:
            # Use a more sophisticated filtering approach
            # First, compute FFT of the signal
            fft_signal = np.fft.rfft(normalized_signal)
            freqs = np.fft.rfftfreq(len(normalized_signal), d=time_range/len(normalized_signal))
            
            # Create a bandpass filter between 0.7 Hz (42 BPM) and 4.0 Hz (240 BPM)
            mask = (freqs >= 0.7) & (freqs <= 4.0)
            
            # Apply filter
            fft_signal[~mask] = 0
            
            # Inverse FFT to get filtered signal
            filtered_signal = np.fft.irfft(fft_signal)
            
            # Apply additional smoothing using a moving average filter
            window_size = 3
            smoothed_signal = np.convolve(filtered_signal, np.ones(window_size)/window_size, mode='valid')
        except Exception as e:
            print(f"Error in signal filtering: {e}")
            # Fallback to simpler filtering
            window_size = 3
            filtered_signal = np.convolve(normalized_signal, np.ones(window_size)/window_size, mode='valid')
            smoothed_signal = filtered_signal
        
        # Improved peak detection using adaptive thresholds
        # Compute a dynamic threshold based on signal characteristics
        mean_signal = np.mean(smoothed_signal)
        std_signal = np.std(smoothed_signal)
        threshold = mean_signal + 0.5 * std_signal
        
        peaks = []
        for i in range(1, len(smoothed_signal)-1):
            # Find peaks by comparing with neighbors and checking against threshold
            if (smoothed_signal[i] > smoothed_signal[i-1] and 
                smoothed_signal[i] > smoothed_signal[i+1] and 
                smoothed_signal[i] > threshold):
                peaks.append(i)
        
        # Calculate BPM from time between peaks with better validation
        if len(peaks) > 2:  # Need at least 3 peaks for reliable calculation
            # Convert peak indices to timestamps
            peak_times = []
            for p in peaks:
                # Adjust index to account for buffer size difference due to filtering
                adjusted_idx = min(p, len(self.timestamps) - 1)
                peak_times.append(self.timestamps[adjusted_idx])
            
            # Calculate intervals between peaks
            intervals = np.diff(peak_times)
            
            # Filter out unreasonable intervals (too short or too long)
            valid_intervals = intervals[(intervals > 0.25) & (intervals < 1.5)]
            
            if len(valid_intervals) > 1:
                # Calculate BPM from average interval
                mean_interval = np.mean(valid_intervals)
                if mean_interval > 0:
                    bpm = 60.0 / mean_interval
                    
                    # Apply physiological constraints (40-200 BPM for humans)
                    if 40 <= bpm <= 200:
                        # Add to BPM history for smoothing with weighted update
                        if hasattr(self, 'last_valid_bpm') and self.last_valid_bpm is not None:
                            # Limit maximum change rate for stability
                            max_change = 10
                            if abs(bpm - self.last_valid_bpm) > max_change:
                                # Limit the change
                                if bpm > self.last_valid_bpm:
                                    bpm = self.last_valid_bpm + max_change
                                else:
                                    bpm = self.last_valid_bpm - max_change
                        
                        # Update last valid BPM
                        self.last_valid_bpm = bpm
                        
                        # Add to BPM history for trend analysis
                        self.bpm_history.append(bpm)
                        # Keep last 10 BPM values for smoothing
                        if len(self.bpm_history) > 10:
                            self.bpm_history = self.bpm_history[-10:]
                        
                        # Calculate HRV (improved method)
                        if len(valid_intervals) > 2:
                            # Calculate RMSSD (root mean square of successive differences)
                            successive_diffs = np.diff(valid_intervals)
                            rmssd = np.sqrt(np.mean(successive_diffs**2)) * 1000  # Convert to ms
                            self.hrv_data.append(rmssd)
                            # Keep last 5 HRV values
                            if len(self.hrv_data) > 5:
                                self.hrv_data = self.hrv_data[-5:]
                        
                        # Return exponentially weighted moving average for stability
                        # Give more weight to recent measurements
                        weights = np.exp(np.linspace(0, 1, len(self.bpm_history)))
                        weighted_bpm = np.average(self.bpm_history, weights=weights)
                        return weighted_bpm
        
        # Fall back to current BPM if calculation failed
        return self.current_bpm
        
    def get_current_bpm(self):
        """Get the current BPM reading from webcam or mock data if unavailable"""
        # Check if we're in an active monitoring session
        if not self.is_processing:
            return self.current_bpm
        
        # If webcam is active, process the frame
        if self.cap is not None and self.is_processing:
            try:
                # Thread-safe frame processing
                if not hasattr(st.session_state, 'frame_processing_active'):
                    st.session_state.frame_processing_active = False
                
                # Only process a new frame if the previous one finished
                if not st.session_state.frame_processing_active:
                    st.session_state.frame_processing_active = True
                    
                    # Read frame from camera 
                    ret, frame = self.cap.read()
                    
                    if ret:
                        # Store frame for preview
                        self.last_frame = frame.copy()
                        
                        # Process frame for heart rate
                        processed_frame, roi = self._extract_forehead_roi(frame)
                        self.forehead_roi = roi
                        
                        # Extract signal from ROI
                        green_val = self._process_roi(frame, roi)
                        
                        # Store signal data with timestamp
                        current_time = time.time()
                        self.signal_buffer.append(green_val)
                        self.timestamps.append(current_time)
                        
                        # Keep buffer at reasonable size
                        if len(self.signal_buffer) > self.buffer_size:
                            self.signal_buffer = self.signal_buffer[-self.buffer_size:]
                            self.timestamps = self.timestamps[-self.buffer_size:]
                        
                        # Calculate BPM every 10 frames or when we have enough samples
                        if len(self.signal_buffer) % 5 == 0 or len(self.signal_buffer) == self.buffer_size:
                            new_bpm = self._calculate_bpm()
                            if 40 <= new_bpm <= 200:  # Validate the BPM is in a reasonable range
                                self.current_bpm = int(new_bpm)
                            self.bpm_ready = True
                    
                    st.session_state.frame_processing_active = False
                
                # Calculate heart rate variability (HRV) if available
                hrv = int(np.mean(self.hrv_data)) if self.hrv_data else np.random.randint(10, 50)
                
                return {
                    'bpm': int(self.current_bpm),
                    'ready': self.bpm_ready,
                    'zone': self._get_heart_rate_zone(self.current_bpm),
                    'hrv': hrv
                }
            except Exception as e:
                print(f"Error processing frame: {e}")
                st.session_state.frame_processing_active = False
        
        # If webcam isn't available or processing failed, use mock data
        # Only update the mock BPM occasionally to make it more stable
        if time.time() - st.session_state.last_update_time >= 1.0:
            self.current_bpm = max(50, min(180, self.current_bpm + np.random.randint(-5, 6)))
        
        return {
            'bpm': int(self.current_bpm),
            'ready': True,
            'zone': self._get_heart_rate_zone(self.current_bpm),
            'hrv': np.random.randint(10, 50)
        }
        
    def get_latest_frame(self):
        """Get the latest processed frame as base64 encoded image"""
        if self.last_frame is not None:
            try:
                # Get camera settings from session state
                settings = st.session_state.get('camera_settings', {
                    'use_face_detection': True,
                    'show_roi': True
                })
                
                # Draw BPM text on frame
                frame = self.last_frame.copy()
                cv2.putText(frame, f"BPM: {self.current_bpm}", (20, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Draw heart rate zone
                zone = self._get_heart_rate_zone(self.current_bpm)
                zone_colors = {
                    'Rest': (0, 255, 0),      # Green
                    'Light': (0, 255, 128),   # Light Green
                    'Moderate': (0, 255, 255),  # Yellow
                    'Vigorous': (0, 128, 255),  # Orange
                    'Maximum': (0, 0, 255)    # Red
                }
                zone_color = zone_colors.get(zone, (0, 255, 0))
                cv2.putText(frame, f"Zone: {zone}", (20, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, zone_color, 2)
                
                # Draw forehead ROI if enabled
                if settings.get('show_roi', True) and self.forehead_roi is not None:
                    x, y, w, h = self.forehead_roi
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Draw signal graph at bottom of frame
                if len(self.signal_buffer) > 5:
                    # Normalize signal for display
                    signal = np.array(self.signal_buffer[-30:])
                    if np.std(signal) > 0:
                        normalized_signal = (signal - np.mean(signal)) / np.std(signal)
                        normalized_signal = normalized_signal * 30 + 200  # Scale and shift
                        
                        # Draw graph background
                        cv2.rectangle(frame, (10, 170), (310, 230), (0, 0, 0), -1)
                        
                        # Draw signal line
                        pts = np.array([[10 + i * 10, int(normalized_signal[i])] 
                                        for i in range(min(30, len(normalized_signal)))], np.int32)
                        pts = pts.reshape((-1, 1, 2))
                        cv2.polylines(frame, [pts], False, (0, 255, 255), 2)
                
                # Convert to RGB for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Convert numpy array to byte string using PIL
                img = Image.fromarray(frame_rgb)
                with io.BytesIO() as buffer:
                    img.save(buffer, format="JPEG")
                    return base64.b64encode(buffer.getvalue()).decode('utf-8')
            except Exception as e:
                print(f"Error creating frame display: {e}")
        
        # If no real camera frame is available, generate a mock frame
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        t = time.time() * 2
        scale = 0.5 + 0.2 * np.sin(t)  # Pulsing effect
        
        # Draw a simple heart shape
        for x in range(320):
            for y in range(240):
                # Normalize coordinates to [-1, 1]
                nx = (x - 160) / 80
                ny = (y - 120) / 80
                
                # Heart curve formula
                heart = (nx*nx + ny*ny - 1)**3 - nx*nx * ny*ny*ny
                
                if heart * scale < 0:
                    # Red color with pulsing intensity
                    intensity = int(200 + 55 * np.sin(t))
                    frame[y, x] = [0, 0, intensity]
        
        # Add text
        cv2.putText(frame, "No camera access", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Mock BPM: {self.current_bpm}", (20, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Convert numpy array to byte string using PIL
        img = Image.fromarray(frame)
        with io.BytesIO() as buffer:
            img.save(buffer, format="JPEG")
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def _get_heart_rate_zone(self, bpm):
        if bpm < 60: return 'Rest'
        elif bpm < 100: return 'Light'
        elif bpm < 140: return 'Moderate'
        elif bpm < 170: return 'Vigorous'
        else: return 'Maximum'
        
    def list_sessions(self):
        # Generate some mock session data
        today = datetime.now()
        yesterday = today - timedelta(days=1)
        
        return [
            {'session_id': today.strftime("%Y%m%d_%H%M%S"), 'start_time': today.isoformat(), 'duration': 300, 'avg_bpm': 75},
            {'session_id': yesterday.strftime("%Y%m%d_%H%M%S"), 'start_time': yesterday.isoformat(), 'duration': 420, 'avg_bpm': 85},
            {'session_id': '20240324_150000', 'start_time': '2024-03-24T15:00:00', 'duration': 360, 'avg_bpm': 92}
        ]
        
    def get_session_data(self, session_id=None):
        # Generate mock session data
        session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        start_time = datetime.now() - timedelta(minutes=5)
        timestamps = [start_time + timedelta(seconds=i*10) for i in range(30)]
        
        # Create more realistic BPM pattern with some variability
        base_bpm = 70 + np.random.randint(-5, 6)
        bpm_values = []
        
        for i in range(30):
            # Add a slight upward trend
            trend = i * 0.2
            # Add some sine wave variation
            variation = 15 * np.sin(i/5)
            # Add some random noise
            noise = np.random.randint(-3, 4)
            
            bpm = base_bpm + trend + variation + noise
            bpm_values.append(int(max(50, min(180, bpm))))
        
        # Calculate zone distribution
        zones = {'Rest': 0, 'Light': 0, 'Moderate': 0, 'Vigorous': 0, 'Maximum': 0}
        for bpm in bpm_values:
            zone = self._get_heart_rate_zone(bpm)
            zones[zone] += 1
        
        # Convert to percentages
        total = len(bpm_values)
        for zone in zones:
            zones[zone] = round(zones[zone] * 100 / total, 1)
        
        return {
            'session_id': session_id,
            'start_time': start_time.isoformat(),
            'end_time': timestamps[-1].isoformat(),
            'bpm_values': bpm_values,
            'timestamps': [ts.isoformat() for ts in timestamps],
            'summary': {
                'avg_bpm': round(sum(bpm_values)/len(bpm_values), 1),
                'min_bpm': min(bpm_values),
                'max_bpm': max(bpm_values),
                'duration': 300,
                'samples': len(bpm_values),
                'zones': zones
            }
        }
        
    def generate_report(self, session_id, format='pdf'):
        # Mock report generation
        if format == 'json':
            return self.get_session_data(session_id)
        else:
            return "PDF report would be generated here"

class SymptomAnalyzer:
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.symptoms = self._load_symptoms()
        
    def _load_symptoms(self):
        # Create default symptom data
        return {
            "headache": {"id": "S1", "name": "Headache", "description": "Pain in the head or upper neck"},
            "fever": {"id": "S2", "name": "Fever", "description": "Elevated body temperature above normal range"},
            "cough": {"id": "S3", "name": "Cough", "description": "Sudden expulsion of air from the lungs"},
            "fatigue": {"id": "S4", "name": "Fatigue", "description": "Extreme tiredness or exhaustion"},
            "nausea": {"id": "S5", "name": "Nausea", "description": "Sensation of unease in the stomach with an urge to vomit"},
            "sore_throat": {"id": "S6", "name": "Sore Throat", "description": "Pain, scratchiness or irritation of the throat"},
            "shortness_of_breath": {"id": "S7", "name": "Shortness of Breath", "description": "Difficulty breathing or dyspnea"},
            "chest_pain": {"id": "S8", "name": "Chest Pain", "description": "Discomfort or pain in the chest area"},
            "abdominal_pain": {"id": "S9", "name": "Abdominal Pain", "description": "Pain felt between the chest and pelvic regions"},
            "muscle_aches": {"id": "S10", "name": "Muscle Aches", "description": "Pain in muscles throughout the body"}
        }
        
    def analyze_symptoms(self, symptoms, user_info=None):
        # Mock analysis with more realistic and varied results based on input symptoms
        conditions = []
        recommendations = []
        
        # Process symptoms to determine conditions
        if not symptoms:
            return {
                'possible_conditions': [],
                'recommendations': ["No symptoms provided for analysis"],
                'disclaimer': 'This is not a substitute for professional medical advice. Please consult a healthcare provider for proper diagnosis and treatment.'
            }
        
        # Convert symptoms to lowercase for matching
        symptoms_lower = [s.lower() for s in symptoms]
        
        # Check for common symptom patterns and create appropriate responses
        if 'fever' in symptoms_lower:
            if 'cough' in symptoms_lower or 'sore throat' in symptoms_lower:
                conditions.append({'name': 'Common Cold', 'probability': 75, 'severity': 1})
                conditions.append({'name': 'Influenza', 'probability': 60, 'severity': 2})
                
                if 'shortness of breath' in symptoms_lower:
                    conditions.append({'name': 'COVID-19', 'probability': 50, 'severity': 3})
                    recommendations.append('Consider getting tested for COVID-19')
                    
            if 'headache' in symptoms_lower or 'muscle aches' in symptoms_lower:
                if not any(c['name'] == 'Influenza' for c in conditions):
                    conditions.append({'name': 'Influenza', 'probability': 65, 'severity': 2})
        
        if 'chest pain' in symptoms_lower:
            if 'shortness of breath' in symptoms_lower:
                conditions.append({'name': 'Possible Cardiovascular Issue', 'probability': 45, 'severity': 4})
                recommendations.append('Seek immediate medical attention for chest pain combined with shortness of breath')
            else:
                conditions.append({'name': 'Angina', 'probability': 30, 'severity': 3})
                recommendations.append('Consult a healthcare provider soon regarding your chest pain')
        
        if 'headache' in symptoms_lower:
            if 'nausea' in symptoms_lower:
                conditions.append({'name': 'Migraine', 'probability': 65, 'severity': 2})
                recommendations.append('Rest in a dark, quiet room and stay hydrated')
                recommendations.append('Over-the-counter pain relievers may help with migraine symptoms')
        
        if 'abdominal pain' in symptoms_lower:
            if 'nausea' in symptoms_lower:
                conditions.append({'name': 'Gastroenteritis', 'probability': 55, 'severity': 2})
                recommendations.append('Stay hydrated and consider a bland diet')
        
        # If no specific conditions matched, provide generic conditions based on symptoms
        if not conditions:
            # Default condition based on number of symptoms
            severity = min(len(symptoms), 3)
            conditions.append({'name': 'General Malaise', 'probability': 80, 'severity': 1})
            if len(symptoms) > 2:
                conditions.append({'name': 'Viral Infection', 'probability': 40, 'severity': 2})
        
        # Add general recommendations if none specific were added
        if not recommendations:
            recommendations.append('Rest and stay hydrated')
            recommendations.append('Take over-the-counter medication for symptom relief if needed')
            
        recommendations.append('If symptoms persist for more than a week, consult a healthcare provider')
        
        # Sort conditions by probability
        conditions.sort(key=lambda x: x['probability'], reverse=True)
        
        return {
            'possible_conditions': conditions,
            'recommendations': recommendations,
            'disclaimer': 'This is not a substitute for professional medical advice. Please consult a healthcare provider for proper diagnosis and treatment.'
        }

# Standalone PDF report generator function
def generate_pdf_report(session_data, symptom_results=None, output_path=None, theme_color="#1E88E5", include_charts=True):
    """Generate a PDF report for health data"""
    # Create a temporary file if no output path provided
    if not output_path:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            output_path = tmp.name
    
    try:
        # Create a simple PDF report
        pdf = FPDF()
        pdf.add_page()
        
        # Header
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, "Health Report", 0, 1, 'C')
        pdf.ln(5)
        
        # Session information
        if session_data:
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, "Heart Rate Data", 0, 1)
            
            pdf.set_font('Arial', '', 12)
            session_id = session_data.get('session_id', 'Unknown')
            pdf.cell(0, 10, f"Session ID: {session_id}", 0, 1)
            
            # Summary stats
            summary = session_data.get('summary', {})
            avg_bpm = summary.get('avg_bpm', 0)
            min_bpm = summary.get('min_bpm', 0)
            max_bpm = summary.get('max_bpm', 0)
            
            pdf.cell(0, 10, f"Average BPM: {avg_bpm}", 0, 1)
            pdf.cell(0, 10, f"Min BPM: {min_bpm}   Max BPM: {max_bpm}", 0, 1)
            pdf.ln(5)
        
        # Symptom results
        if symptom_results:
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, "Symptom Analysis", 0, 1)
            
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, "Possible Conditions:", 0, 1)
            
            pdf.set_font('Arial', '', 12)
            conditions = symptom_results.get('possible_conditions', [])
            for condition in conditions:
                name = condition.get('name', 'Unknown')
                probability = condition.get('probability', 0)
                severity = condition.get('severity', 1)
                
                pdf.cell(0, 10, f"- {name} ({probability}% probability, Severity: {severity})", 0, 1)
            
            pdf.ln(5)
            
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, "Recommendations:", 0, 1)
            
            pdf.set_font('Arial', '', 12)
            recommendations = symptom_results.get('recommendations', [])
            for rec in recommendations:
                pdf.cell(0, 10, f"- {rec}", 0, 1)
            
            pdf.ln(5)
        
        # Disclaimer
        pdf.set_font('Arial', 'I', 10)
        pdf.multi_cell(0, 10, "Disclaimer: This report is for informational purposes only and is not a substitute for professional medical advice.")
        
        # Save the PDF
        pdf.output(output_path)
        return output_path
        
    except Exception as e:
        # If PDF generation fails, create a simple text file
        with open(output_path, 'w') as f:
            f.write(f"Error generating report: {str(e)}")
        return output_path

# Function to initialize data directories
def init_data_directories():
    """Initialize necessary data directories"""
    base_dir = os.getcwd()
    data_dirs = [
        os.path.join(base_dir, 'data'),
        os.path.join(base_dir, 'data', 'bpm_sessions'),
        os.path.join(base_dir, 'data', 'medical_data'),
        os.path.join(base_dir, 'data', 'reports'),
        os.path.join(base_dir, 'data', 'analysis')
    ]
    for dir_path in data_dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    # Create default data files if they don't exist
    symptoms_file = os.path.join(base_dir, 'data', 'symptoms.json')
    conditions_file = os.path.join(base_dir, 'data', 'conditions.json')
    mapping_file = os.path.join(base_dir, 'data', 'symptom_condition_mapping.json')
    
    if not os.path.exists(symptoms_file):
        with open(symptoms_file, 'w') as f:
            json.dump({
                "headache": {"name": "Headache", "description": "Pain in the head"},
                "fever": {"name": "Fever", "description": "Elevated body temperature"},
                "cough": {"name": "Cough", "description": "Sudden expulsion of air"}
            }, f, indent=2)
            
    if not os.path.exists(conditions_file):
        with open(conditions_file, 'w') as f:
            json.dump({
                "common_cold": {"name": "Common Cold", "severity": 1},
                "flu": {"name": "Influenza", "severity": 2},
                "covid": {"name": "COVID-19", "severity": 3}
            }, f, indent=2)
            
    if not os.path.exists(mapping_file):
        with open(mapping_file, 'w') as f:
            json.dump({
                "headache": ["common_cold", "flu", "covid"],
                "fever": ["flu", "covid"],
                "cough": ["common_cold", "flu", "covid"]
            }, f, indent=2)
    
    return os.path.join(base_dir, 'data')

# Configure the Streamlit page
st.set_page_config(
    page_title="HealthAssist AI Dashboard",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem !important;
        font-weight: 600 !important;
        color: #E53935 !important;
        margin-bottom: 1rem !important;
    }
    .sub-header {
        font-size: 1.8rem !important;
        font-weight: 500 !important;
        color: #1E88E5 !important;
        margin-top: 1rem !important;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 600;
        margin: 0;
    }
    .metric-label {
        font-size: 1rem;
        color: #6c757d;
        margin: 0;
    }
    .heart-rate-zone {
        padding: 0.5rem 1rem;
        border-radius: 0.25rem;
        color: white;
        font-weight: 600;
        text-align: center;
    }
    .report-card {
        border: 1px solid #e9ecef;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize data directory
data_dir = init_data_directories()

# Initialize session state variables
if 'bpm_monitor' not in st.session_state:
    st.session_state.bpm_monitor = BPMMonitor(data_dir=os.path.join(data_dir, 'bpm_sessions'))
if 'symptom_analyzer' not in st.session_state:
    st.session_state.symptom_analyzer = SymptomAnalyzer(data_path=os.path.join(data_dir, 'medical_data'))
if 'monitoring_active' not in st.session_state:
    st.session_state.monitoring_active = False
if 'current_bpm' not in st.session_state:
    st.session_state.current_bpm = 0
if 'symptom_results' not in st.session_state:
    st.session_state.symptom_results = None
if 'current_report' not in st.session_state:
    st.session_state.current_report = None
if 'session_history' not in st.session_state:
    st.session_state.session_history = []
if 'camera_settings' not in st.session_state:
    st.session_state.camera_settings = {
        'device_index': 0,
        'use_face_detection': True,
        'show_roi': True
    }
if 'user_info' not in st.session_state:
    st.session_state.user_info = {
        'age': 35,
        'gender': 'male',
        'name': 'User'
    }
if 'last_update_time' not in st.session_state:
    st.session_state.last_update_time = time.time()
if 'frame_processing_active' not in st.session_state:
    st.session_state.frame_processing_active = False
if 'update_interval' not in st.session_state:
    st.session_state.update_interval = 0.2  # Update every 200ms by default

# Sidebar for application navigation
with st.sidebar:
    st.markdown("<h1 style='text-align: center;'>HealthAssist AI</h1>", unsafe_allow_html=True)
    
    # Fix for image error - create a default logo instead of using external file
    try:
        # Try to load logo if it exists
        if os.path.exists("app/static/images/health-logo.png"):
            st.image("app/static/images/health-logo.png", use_column_width=True)
        else:
            # Create a placeholder logo using matplotlib
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.set_facecolor('#f0f2f6')
            
            # Draw a heart shape
            t = np.linspace(0, 2*np.pi, 100)
            x = 16 * np.sin(t)**3
            y = 13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)
            
            ax.plot(x, y, color='#E53935', linewidth=3)
            ax.fill(x, y, color='#E53935', alpha=0.7)
            ax.text(0, 0, "HealthAssist AI", ha='center', va='center', fontsize=10, 
                    fontweight='bold', color='white')
            
            ax.set_xlim(-18, 18)
            ax.set_ylim(-15, 15)
            ax.axis('off')
            fig.tight_layout()
            
            # Display the generated logo
            st.pyplot(fig)
            plt.close(fig)
    except Exception as e:
        st.write("HealthAssist AI")  # Fallback if image creation fails
    
    # User info
    st.markdown("### User Profile")
    if st.session_state.user_info:
        st.markdown(f"**Name:** {st.session_state.user_info['name']}")
        st.markdown(f"**Age:** {st.session_state.user_info['age']}")
        st.markdown(f"**Gender:** {st.session_state.user_info['gender'].capitalize()}")
    
    # Medical Disclaimer
    st.markdown("---")
    st.markdown("""
    **Medical Disclaimer:** This application is for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment.
    """)

# App title and description
st.markdown("<h1 class='main-header'>HealthAssist AI Dashboard</h1>", unsafe_allow_html=True)
st.markdown("""
This dashboard provides comprehensive health monitoring and analysis tools
to help you track your vital signs and symptoms.
""")

# Create tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["Heart Rate Monitor", "Symptom Analyzer", "Health Reports"])

# Heart Rate Monitor Tab
with tab1:
    st.markdown("<h2 class='sub-header'>Heart Rate Monitor</h2>", unsafe_allow_html=True)
    
    # Camera access notice with troubleshooting
    st.info("""
    📹 **Camera Access Required:** This feature uses your webcam to calculate heart rate by detecting subtle color changes in your skin.
    Please allow camera access when prompted.

    **For best results:**
    - Ensure good lighting on your face
    - Stay relatively still while monitoring
    - Position your face clearly in the camera view
    """)

    # Troubleshooting expander
    with st.expander("Camera Troubleshooting"):
        st.markdown("""
        **If the camera doesn't activate:**
        1. Make sure your browser has permission to access the camera
        2. Try selecting a different camera from the dropdown
        3. Try refreshing the page
        4. If using Windows, check if another application is using the camera
        5. If all else fails, the app will use simulated data for demonstration
        """)
        
        # Camera test button
        if st.button("Test Camera Access"):
            try:
                cap = cv2.VideoCapture(0)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        st.success("✅ Camera is working! You can now start heart rate monitoring.")
                        # Show test frame
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        st.image(frame_rgb, caption="Camera Test Image", width=320)
                    else:
                        st.error("❌ Camera opened but couldn't read frame. Please check if another application is using it.")
                else:
                    st.error("❌ Couldn't access the camera. Please check your permissions and try again.")
                # Always release the camera
                cap.release()
            except Exception as e:
                st.error(f"❌ Error accessing camera: {e}")
                st.info("The app will use simulated data for demonstration.")
    
    col1, col2 = st.columns([2, 3])
    
    with col1:
        # Heart rate control panel
        st.markdown("### Control Panel")
        
        # Camera selection
        camera_devices = [0, 1, 2]  # Common camera indices
        selected_camera = st.selectbox(
            "Select Camera", 
            options=camera_devices,
            format_func=lambda x: f"Camera {x}" + (" (Default)" if x == 0 else ""),
            index=0,
            help="Select which camera to use for heart rate monitoring"
        )
        
        # Camera settings
        with st.expander("Camera Settings", expanded=False):
            use_face_detection = st.checkbox("Use Face Detection", value=True, 
                                             help="Detect face to improve heart rate measurement")
            show_roi = st.checkbox("Show ROI", value=True, 
                                   help="Show region of interest used for heart rate calculation")
            
            # Add frame rate control slider
            refresh_rate = st.slider(
                "Refresh Rate", 
                min_value=1, 
                max_value=60, 
                value=int(1/st.session_state.update_interval) if st.session_state.update_interval > 0 else 5,
                help="How many times per second to update the display (higher values may cause performance issues)"
            )
            # Update the session state interval based on slider
            st.session_state.update_interval = 1.0 / refresh_rate
        
        # Camera preview
        camera_placeholder = st.empty()
        
        if st.session_state.monitoring_active:
            # Get latest frame if available
            frame_data = st.session_state.bpm_monitor.get_latest_frame()
            if frame_data:
                camera_placeholder.markdown(
                    f"""
                    <div style="display: flex; justify-content: center; margin-bottom: 1rem;">
                        <img src="data:image/jpeg;base64,{frame_data}" width="320" height="240" style="border-radius: 0.5rem; border: 1px solid #ddd;">
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                camera_placeholder.markdown(
                    """
                    <div style="display: flex; justify-content: center; align-items: center; width: 320px; height: 240px; background: #f0f0f0; margin-bottom: 1rem; border-radius: 0.5rem; border: 1px solid #ddd;">
                        <p style="text-align: center; color: #666;">Camera feed unavailable</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            camera_placeholder.markdown(
                """
                <div style="display: flex; justify-content: center; align-items: center; width: 320px; height: 240px; background: #f0f0f0; margin-bottom: 1rem; border-radius: 0.5rem; border: 1px solid #ddd;">
                    <p style="text-align: center; color: #666;">Start monitoring to activate camera</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Monitoring controls
        monitor_col1, monitor_col2 = st.columns(2)
        
        with monitor_col1:
            if not st.session_state.monitoring_active:
                if st.button("Start Monitoring", key="start_btn", use_container_width=True):
                    # Store camera settings in session state
                    st.session_state.camera_settings = {
                        'device_index': selected_camera,
                        'use_face_detection': use_face_detection,
                        'show_roi': show_roi
                    }
                    # Start monitoring with the selected camera
                    st.session_state.bpm_monitor.start_monitoring(record_data=True, camera_index=selected_camera)
                    st.session_state.monitoring_active = True
                    st.rerun()
        
        with monitor_col2:
            if st.session_state.monitoring_active:
                if st.button("Stop Monitoring", key="stop_btn", type="primary", use_container_width=True):
                    st.session_state.bpm_monitor.stop_monitoring()
                    st.session_state.monitoring_active = False
                    st.rerun()
        
        # Current BPM display
        if st.session_state.monitoring_active:
            bpm_info = st.session_state.bpm_monitor.get_current_bpm()
            st.session_state.current_bpm = bpm_info['bpm']
            
            # BPM gauge
            st.markdown(f"""
            <div style="text-align: center; margin: 1.5rem 0;">
                <h1 style="font-size: 4rem; margin-bottom: 0;">{bpm_info['bpm']}</h1>
                <p style="font-size: 1.5rem; margin-top: 0;">BPM</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Heart rate zone
            zone = bpm_info['zone']
            zone_colors = {
                'Rest': '#4CAF50',      # Green
                'Light': '#8BC34A',     # Light Green
                'Moderate': '#FFC107',  # Amber
                'Vigorous': '#FF9800',  # Orange
                'Maximum': '#F44336'    # Red
            }
            zone_color = zone_colors.get(zone, '#4CAF50')
            
            st.markdown(f"""
            <div style="text-align: center; margin: 1rem 0;">
                <div class="heart-rate-zone" style="background-color: {zone_color};">
                    <p style="margin: 0; font-weight: bold;">Zone: {zone}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # HRV display and metrics
            metric_col1, metric_col2 = st.columns(2)
            
            with metric_col1:
                st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-value">{bpm_info['hrv']}</p>
                    <p class="metric-label">HRV (ms)</p>
                </div>
                """, unsafe_allow_html=True)
            
            with metric_col2:
                # Calculate session duration if available
                duration = "0:00"
                if hasattr(st.session_state.bpm_monitor, 'session_start_time') and st.session_state.bpm_monitor.session_start_time:
                    elapsed = datetime.now() - st.session_state.bpm_monitor.session_start_time
                    minutes, seconds = divmod(elapsed.seconds, 60)
                    duration = f"{minutes}:{seconds:02d}"
                
                st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-value">{duration}</p>
                    <p class="metric-label">Duration</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Status message
            st.success("Monitoring active" if st.session_state.monitoring_active else "Monitoring inactive")
        else:
            st.info("Click 'Start Monitoring' to begin heart rate analysis")
    
    with col2:
        # Heart rate visualization
        st.markdown("### Heart Rate Visualization")
        
        # Customization options
        custom_col1, custom_col2, custom_col3 = st.columns(3)
        
        with custom_col1:
            time_range = st.selectbox("Time Window", ["30 sec", "1 min", "2 min", "5 min"], index=1)
        
        with custom_col2:
            line_color = st.color_picker("Line Color", "#1E88E5")
        
        with custom_col3:
            line_thickness = st.slider("Line Thickness", 1, 5, 2)
        
        # Convert time range to seconds
        if time_range == "30 sec":
            max_points = 30
        elif time_range == "1 min":
            max_points = 60
        elif time_range == "2 min":
            max_points = 120
                        else:
            max_points = 300
        
        # Create a placeholder for the chart
        chart_placeholder = st.empty()
        
        if st.session_state.monitoring_active:
            # Create or update chart data
            if 'bpm_chart_data' not in st.session_state:
                st.session_state.bpm_chart_data = {
                    'time': [0],
                    'bpm': [st.session_state.current_bpm]
                }
                st.session_state.chart_start_time = time.time()
                    else:
                # Add new data point
                current_time = time.time() - st.session_state.chart_start_time
                st.session_state.bpm_chart_data['time'].append(current_time)
                st.session_state.bpm_chart_data['bpm'].append(st.session_state.current_bpm)
                
                # Keep only the last max_points data points
                if len(st.session_state.bpm_chart_data['time']) > max_points:
                    st.session_state.bpm_chart_data['time'] = st.session_state.bpm_chart_data['time'][-max_points:]
                    st.session_state.bpm_chart_data['bpm'] = st.session_state.bpm_chart_data['bpm'][-max_points:]
            
            # Convert to DataFrame for plotting
            df = pd.DataFrame(st.session_state.bpm_chart_data)
            
            # Create chart
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(df['time'], df['bpm'], color=line_color, linewidth=line_thickness)
            
            # Add heart rate zones
            ax.axhspan(0, 60, alpha=0.2, color='#4CAF50', label='Rest')
            ax.axhspan(60, 100, alpha=0.2, color='#8BC34A', label='Light')
            ax.axhspan(100, 140, alpha=0.2, color='#FFC107', label='Moderate')
            ax.axhspan(140, 170, alpha=0.2, color='#FF9800', label='Vigorous')
            ax.axhspan(170, 200, alpha=0.2, color='#F44336', label='Maximum')
            
            # Set chart properties
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('BPM')
            ax.set_title('Real-time Heart Rate Monitoring')
            ax.set_ylim(40, 180)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')
            
            # Display chart in placeholder
            chart_placeholder.pyplot(fig)
            plt.close(fig)
                else:
            chart_placeholder.info("Start monitoring to see real-time heart rate data")
        
        # Display heart rate zone information
        st.markdown("### Heart Rate Zones")
        
        zone_info = [
            {"zone": "Rest", "range": "Under 60 BPM", "color": "#4CAF50", "description": "Heart at rest, typically during sleep or deep relaxation."},
            {"zone": "Light", "range": "60-100 BPM", "color": "#8BC34A", "description": "Normal resting heart rate for adults. Light daily activities."},
            {"zone": "Moderate", "range": "100-140 BPM", "color": "#FFC107", "description": "Moderate exercise intensity. Good for improving cardiovascular health."},
            {"zone": "Vigorous", "range": "140-170 BPM", "color": "#FF9800", "description": "High intensity exercise. Improves cardiorespiratory fitness."},
            {"zone": "Maximum", "range": "170+ BPM", "color": "#F44336", "description": "Maximum effort, not sustainable for long periods. Anaerobic zone."}
        ]
        
        # Display zone information as a table
        zone_df = pd.DataFrame(zone_info)
        st.dataframe(zone_df, hide_index=True, use_container_width=True)
        
        # Automatic refresh section
        if st.session_state.monitoring_active:
            # Instead of using sleep, we'll use a timestamp-based approach
            if 'last_update_time' not in st.session_state:
                st.session_state.last_update_time = time.time()
            
            current_time = time.time()
            # Only rerun if enough time has passed (based on update interval)
            if current_time - st.session_state.last_update_time >= st.session_state.update_interval:
                st.session_state.last_update_time = current_time
                st.rerun()

# Symptom Analyzer Tab
with tab2:
    st.markdown("<h2 class='sub-header'>Symptom Analyzer</h2>", unsafe_allow_html=True)
    
    # Symptom Analysis interface
    symptom_col1, symptom_col2 = st.columns([3, 2])
    
    with symptom_col1:
        st.markdown("### Input Your Symptoms")
        
        # Common symptoms selector with checkboxes for better UX
        st.markdown("**Select common symptoms:**")
        symptom_cols = st.columns(3)
        
        common_symptoms = [
            "Headache", "Fever", "Cough", 
            "Fatigue", "Sore Throat", "Nausea", 
            "Shortness of Breath", "Dizziness", "Chest Pain"
        ]
        
        selected_symptoms = []
        
        # Display symptoms as checkboxes in columns
        for i, symptom in enumerate(common_symptoms):
            col_index = i % 3
            with symptom_cols[col_index]:
                if st.checkbox(symptom, key=f"checkbox_{symptom}"):
                    selected_symptoms.append(symptom)
        
        # Text input for additional symptoms
        st.markdown("**Additional symptoms:**")
        custom_symptoms = st.text_area(
            "Enter additional symptoms, separated by commas",
            placeholder="e.g., joint pain, rash, blurred vision",
            help="Enter any symptoms not listed above"
        )
        
        # Process custom symptoms
        if custom_symptoms:
            additional_symptoms = [s.strip() for s in custom_symptoms.split(',') if s.strip()]
            selected_symptoms.extend(additional_symptoms)
        
        # User information form
        st.markdown("### Personal Information")
        
        user_info_col1, user_info_col2 = st.columns(2)
        
        with user_info_col1:
            age = st.number_input("Age", min_value=1, max_value=120, value=st.session_state.user_info['age'])
        
        with user_info_col2:
            gender = st.selectbox("Gender", ["Male", "Female", "Other"], 
                                index=0 if st.session_state.user_info['gender'].lower() == 'male' else 
                                        1 if st.session_state.user_info['gender'].lower() == 'female' else 2)
        
        # Analyze button
        if st.button("Analyze Symptoms", use_container_width=True, type="primary", disabled=len(selected_symptoms) == 0):
            if len(selected_symptoms) > 0:
                # Update user info in session state
                st.session_state.user_info['age'] = age
                st.session_state.user_info['gender'] = gender.lower()
                
                # Create user info dictionary
                user_info = {
                    "age": age,
                    "gender": gender.lower()
                }
                
                # Analyze symptoms
                with st.spinner("Analyzing symptoms..."):
                    results = st.session_state.symptom_analyzer.analyze_symptoms(selected_symptoms, user_info)
                    st.session_state.symptom_results = results
                    st.rerun()
        else:
                st.error("Please select at least one symptom")
    
    with symptom_col2:
        st.markdown("### Analysis Results")
        
        if st.session_state.symptom_results:
            # Display selected symptoms
            st.markdown("**Selected Symptoms:**")
            symptom_tags = ", ".join(selected_symptoms) if 'selected_symptoms' in locals() and selected_symptoms else "None"
            st.markdown(f"_{symptom_tags}_")
            
            # Possible conditions
            conditions = st.session_state.symptom_results.get('possible_conditions', [])
            if conditions:
                st.markdown("**Possible Conditions:**")
                
                for i, condition in enumerate(conditions):
                    condition_name = condition.get('name', condition.get('condition', 'Unknown'))
                    probability = condition.get('probability', 0)
                    severity = condition.get('severity', 1)
                    
                    # Create condition card
                    severity_color = "#4CAF50" if severity == 1 else "#FFC107" if severity == 2 else "#F44336"
                    
                    st.markdown(f"""
                    <div class="report-card">
                        <h4 style="margin-top: 0;">{condition_name}</h4>
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div style="flex: 1;">
                                <div style="background-color: #e9ecef; border-radius: 0.25rem; height: 0.5rem; width: 100%;">
                                    <div style="background-color: {severity_color}; width: {probability}%; height: 100%; border-radius: 0.25rem;"></div>
            </div>
                                <p style="margin: 0.25rem 0 0 0; text-align: right;">{probability}%</p>
            </div>
                            <div style="margin-left: 1rem;">
                                <span style="color: {severity_color}; font-weight: 600;">{"⚠️ " * severity}</span>
                            </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
            else:
                st.info("No conditions identified from the provided symptoms")
            
            # Recommendations
            recommendations = st.session_state.symptom_results.get('recommendations', [])
            if recommendations:
                st.markdown("**Recommendations:**")
                for rec in recommendations:
                    st.markdown(f"- {rec}")
        else:
            st.info("Select symptoms and click 'Analyze Symptoms' to see results")
        
        # Medical disclaimer
        st.markdown("---")
        disclaimer = """
        **Medical Disclaimer:** This symptom analysis is for informational purposes only and should not be used for self-diagnosis. Always consult a healthcare provider for proper diagnosis and treatment.
        """
        st.markdown(disclaimer)

# Health Reports Tab
with tab3:
    st.markdown("<h2 class='sub-header'>Health Reports</h2>", unsafe_allow_html=True)
    
    tab3_col1, tab3_col2 = st.columns([1, 2])
    
    with tab3_col1:
        st.markdown("### Session History")
        
        # Get list of recorded sessions
        sessions = st.session_state.bpm_monitor.list_sessions()
        
        if sessions:
            # Update session history in session state
            st.session_state.session_history = sessions
            
            # Create a selection widget for sessions
            session_options = [
                f"{s['session_id']} - {datetime.fromisoformat(s['start_time']).strftime('%d %b %Y, %H:%M') if 'start_time' in s and s['start_time'] != 'Unknown' else 'Unknown'}" 
                for s in sessions
            ]
            
            selected_session = st.selectbox(
                "Select a session to view", 
                session_options,
                index=0
            )
            
            if selected_session:
                # Extract session ID from selection
                session_id = selected_session.split(' - ')[0]
                
                # Session details card
                st.markdown("### Session Details")
                selected_session_data = next((s for s in sessions if s['session_id'] == session_id), None)
                
                if selected_session_data:
                    start_time = datetime.fromisoformat(selected_session_data.get('start_time', datetime.now().isoformat()))
                    duration_sec = selected_session_data.get('duration', 0)
                    minutes, seconds = divmod(duration_sec, 60)
                    duration_str = f"{minutes}:{seconds:02d}"
                    
                    # Display session details in a styled card
                    st.markdown(f"""
                    <div class="report-card">
                        <p><strong>Date:</strong> {start_time.strftime('%d %b %Y')}</p>
                        <p><strong>Time:</strong> {start_time.strftime('%H:%M:%S')}</p>
                        <p><strong>Duration:</strong> {duration_str}</p>
                        <p><strong>Average BPM:</strong> {selected_session_data.get('avg_bpm', 0)}</p>
        </div>
        """, unsafe_allow_html=True)

                # Combine with symptom results if available
                combine_with_symptoms = st.checkbox("Include symptom analysis", value=True if st.session_state.symptom_results else False)
                
                # Report generation options
                st.markdown("### Report Options")
                
                report_format = st.selectbox("Report Format", ["PDF", "JSON"], index=0)
                
                # Theme color selection for PDF reports
                if report_format == "PDF":
                    theme_color = st.color_picker("Report Accent Color", "#1E88E5")
                    include_charts = st.checkbox("Include Charts & Graphs", value=True)
    else:
                    theme_color = "#1E88E5"
                    include_charts = True
                
                # Generate report button
                if st.button("Generate Report", use_container_width=True, type="primary"):
                    with st.spinner("Generating report..."):
                        # Get full session data
                        session_data = st.session_state.bpm_monitor.get_session_data(session_id)
                        
                        # For PDF format
                        if report_format == "PDF":
                            # Create a temporary file
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                                tmp_path = tmp.name
                            
                            # Generate PDF report
                            try:
                                report_path = generate_pdf_report(
                                    session_data=session_data,
                                    symptom_results=st.session_state.symptom_results if combine_with_symptoms else None,
                                    output_path=tmp_path,
                                    theme_color=theme_color,
                                    include_charts=include_charts
                                )
                                
                                # Read the generated PDF
                                with open(report_path, 'rb') as f:
                                    pdf_bytes = f.read()
                                
                                # Store in session state
                                st.session_state.current_report = {
                                    'session_id': session_id,
                                    'format': 'PDF',
                                    'data': pdf_bytes,
                                    'file_path': report_path
                                }
                            except Exception as e:
                                st.error(f"Error generating PDF report: {e}")
                        else:
                            # For JSON format
                            report_data = session_data
                            
                            # Add symptom results if requested
                            if combine_with_symptoms and st.session_state.symptom_results:
                                report_data['symptom_analysis'] = st.session_state.symptom_results
                            
                            # Store in session state
                            st.session_state.current_report = {
                                'session_id': session_id,
                                'format': 'JSON',
                                'data': report_data
                            }
                        
                        st.success(f"Report generated for session {session_id}")
                
                # Download report button if report exists
                if 'current_report' in st.session_state and st.session_state.current_report:
                    # For JSON format
                    if st.session_state.current_report['format'] == "JSON":
                        report_json = json.dumps(st.session_state.current_report['data'], indent=4)
                        st.download_button(
                            label="Download JSON Report",
                            data=report_json,
                            file_name=f"health_report_{session_id}.json",
                            mime="application/json",
                            use_container_width=True
                        )
                    # For PDF format
                    else:
                        st.download_button(
                            label="Download PDF Report",
                            data=st.session_state.current_report['data'],
                            file_name=f"health_report_{session_id}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
        else:
            st.info("No recorded sessions found. Start heart rate monitoring to create a session.")
    
    with tab3_col2:
        st.markdown("### Report Preview")
        
        if 'current_report' in st.session_state and st.session_state.current_report:
            # If PDF format, display a preview image
            if st.session_state.current_report['format'] == 'PDF':
                st.markdown("PDF reports can be downloaded for viewing.")
                
                # Show report details
                st.markdown("**Report Summary:**")
                report_id = st.session_state.current_report['session_id']
                st.markdown(f"- **Session ID:** {report_id}")
                st.markdown(f"- **Format:** {st.session_state.current_report['format']}")
                st.markdown(f"- **Generated:** {datetime.now().strftime('%d %b %Y %H:%M:%S')}")
            
            # If JSON format, display the structured data
else:
                report_data = st.session_state.current_report['data']
                
                # Get BPM values and timestamps
                bpm_values = report_data.get('bpm_values', [])
                timestamps = report_data.get('timestamps', [])
                
                if bpm_values and timestamps:
                    # Convert timestamps to datetime
                    try:
                        time_values = [datetime.fromisoformat(ts) for ts in timestamps]
                    except ValueError:
                        time_values = list(range(len(bpm_values)))
                    
                    # Plot BPM chart
                    st.markdown("#### Heart Rate Session Data")
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(time_values, bpm_values, color='#1E88E5', linewidth=2)
                    
                    # Add heart rate zones
                    ax.axhspan(0, 60, alpha=0.2, color='#4CAF50')
                    ax.axhspan(60, 100, alpha=0.2, color='#8BC34A')
                    ax.axhspan(100, 140, alpha=0.2, color='#FFC107')
                    ax.axhspan(140, 170, alpha=0.2, color='#FF9800')
                    ax.axhspan(170, 200, alpha=0.2, color='#F44336')
                    
                    # Set chart properties
                    ax.set_xlabel('Time')
                    ax.set_ylabel('BPM')
                    ax.set_title(f"Heart Rate Session: {report_data.get('session_id', 'Unknown')}")
                    ax.set_ylim(40, 180)
                    ax.grid(True, alpha=0.3)
                    
                    # Format x-axis labels
                    plt.xticks(rotation=45)
                    fig.tight_layout()
                    
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    # Summary statistics
                    st.markdown("#### Summary Statistics")
                    summary = report_data.get('summary', {})
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Average BPM", summary.get('avg_bpm', 0))
                    with col2:
                        st.metric("Minimum BPM", summary.get('min_bpm', 0))
                    with col3:
                        st.metric("Maximum BPM", summary.get('max_bpm', 0))
                    
                    # Heart rate zone distribution
                    st.markdown("#### Heart Rate Zone Distribution")
                    zones = summary.get('zones', {})
                    if zones:
                        zone_names = list(zones.keys())
                        zone_values = list(zones.values())
                        
                        fig, ax = plt.subplots(figsize=(10, 5))
                        bars = ax.bar(zone_names, zone_values, color=['#4CAF50', '#8BC34A', '#FFC107', '#FF9800', '#F44336'])
                        
                        # Add percentage labels on top of bars
                        for bar in bars:
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                                    f'{height}%', ha='center', va='bottom')
                        
                        ax.set_ylim(0, 100)
                        ax.set_ylabel('Percentage of Time (%)')
                        ax.set_title('Time Spent in Each Heart Rate Zone')
                        
                        st.pyplot(fig)
                        plt.close(fig)
                    
                    # Display symptom analysis if available
                    if 'symptom_analysis' in report_data and report_data['symptom_analysis']:
                        st.markdown("#### Symptom Analysis")
                        symptom_results = report_data['symptom_analysis']
                        
                        # Possible conditions
                        conditions = symptom_results.get('possible_conditions', [])
                        if conditions:
                            condition_data = [
                                {
                                    "Condition": c.get('name', c.get('condition', 'Unknown')),
                                    "Probability": f"{c.get('probability', 0)}%",
                                    "Severity": "⚠️" * c.get('severity', 1)
                                }
                                for c in conditions
                            ]
                            st.table(pd.DataFrame(condition_data))
                        
                        # Recommendations
                        recommendations = symptom_results.get('recommendations', [])
                        if recommendations:
                            st.markdown("**Recommendations:**")
                            for rec in recommendations:
                                st.markdown(f"- {rec}")
                else:
                    st.warning("No heart rate data available for this session")
        else:
            st.info("Select a session and generate a report to see a preview here")

# Footer
st.markdown("---")
st.markdown("© 2024 HealthAssist AI. This application is for informational purposes only and is not a substitute for professional medical advice.")