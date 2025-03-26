import streamlit as st
import cv2
import dlib
import numpy as np
import time
from PIL import Image
import threading
import queue

# Set page config
st.set_page_config(
    page_title="Heart Rate Monitor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #d90429;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #333;
        margin-bottom: 1rem;
    }
    .bpm-display {
        font-size: 3.5rem;
        font-weight: bold;
        color: #d90429;
        margin: 1rem 0;
    }
    .bpm-label {
        font-size: 1rem;
        color: #6c757d;
    }
    .bpm-status {
        font-size: 1rem;
        padding: 0.3rem 0.6rem;
        border-radius: 0.25rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stButton button {
        width: 100%;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Global variables
class HeartRateMonitor:
    def __init__(self):
        # Parameters
        self.width, self.height = 640, 480
        self.video_width, self.video_height = 160, 120
        self.video_channels = 3
        self.levels = 3
        self.alpha = 170
        self.min_frequency = 1.0
        self.max_frequency = 2.0
        self.buffer_size = 150
        self.buffer_index = 0
        self.fps = 15
        self.bpm_calculation_frequency = 10
        self.bpm_buffer_index = 0
        self.bpm_buffer_size = 10
        self.bpm_buffer = np.zeros((self.bpm_buffer_size))
        
        # State
        self.current_bpm = 0
        self.bpm_ready = False
        self.is_monitoring = False
        self.frame_queue = queue.Queue(maxsize=5)
        
        # Initialize face detector
        self.detector = dlib.get_frontal_face_detector()
        try:
            self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
            self.use_landmarks = True
        except:
            self.use_landmarks = False
            st.warning("Shape predictor file not found. Using basic face detection.")
    
    def start_monitoring(self):
        if self.is_monitoring:
            return
        
        # Reset state
        self.bpm_buffer = np.zeros((self.bpm_buffer_size))
        self.bpm_buffer_index = 0
        self.buffer_index = 0
        self.bpm_ready = False
        self.current_bpm = 0
        
        # Start processing
        self.is_monitoring = True
        self.process_thread = threading.Thread(target=self._process_frames)
        self.process_thread.daemon = True
        self.process_thread.start()
    
    def stop_monitoring(self):
        self.is_monitoring = False
        # Wait for any frame processing to finish
        time.sleep(0.5)
    
    def _process_frames(self):
        # Initialize webcam
        webcam = cv2.VideoCapture(0)
        webcam.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
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
        buffer_index = 0
        ptime = time.time()
        
        try:
            while self.is_monitoring:
                # Maintain frame rate
                current_time = time.time()
                elapsed = current_time - ptime
                if elapsed < 1.0/self.fps:
                    time.sleep(1.0/self.fps - elapsed)
                
                # Capture frame
                ret, frame = webcam.read()
                if not ret:
                    st.error("Failed to capture frame from webcam")
                    break
                
                # Calculate FPS
                ctime = time.time()
                fps = 1 / (ctime - ptime)
                ptime = ctime
                
                # Make a copy for display
                display_frame = frame.copy()
                
                # Ensure correct format
                if frame.dtype != np.uint8:
                    frame = frame.astype(np.uint8)
                
                # Convert to grayscale for face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = self.detector(gray)
                
                # Draw FPS
                cv2.putText(display_frame, f'FPS: {int(fps)}', (10, self.height - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if faces:
                    # Process first face found
                    face = faces[0]
                    
                    # Get face bounds
                    x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
                    
                    # Draw rectangle around face
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    
                    # Extract and resize face region
                    detection_frame = frame[y1:y2, x1:x2]
                    if detection_frame.size == 0:
                        continue
                    
                    detection_frame = cv2.resize(detection_frame, (self.video_width, self.video_height))
                    
                    # Apply Eulerian Video Magnification
                    video_gauss[buffer_index] = self._build_gauss(detection_frame, self.levels + 1)[self.levels]
                    fourier_transform = np.fft.fft(video_gauss, axis=0)
                    
                    # Bandpass filter
                    fourier_transform[mask == False] = 0
                    
                    # Calculate heart rate
                    if buffer_index % self.bpm_calculation_frequency == 0:
                        i += 1
                        for buf in range(self.buffer_size):
                            fourier_transform_avg[buf] = np.real(fourier_transform[buf]).mean()
                        
                        # Find frequency with highest magnitude
                        hz = frequencies[np.argmax(fourier_transform_avg)]
                        bpm = 60.0 * hz
                        
                        # Apply constraints
                        if 50 <= bpm <= 180:  # Normal human heart rate range
                            self.bpm_buffer[self.bpm_buffer_index] = bpm
                            self.bpm_buffer_index = (self.bpm_buffer_index + 1) % self.bpm_buffer_size
                    
                    # Amplify and reconstruct
                    filtered = np.real(np.fft.ifft(fourier_transform, axis=0)) * self.alpha
                    filtered_frame = self._reconstruct_frame(filtered, buffer_index, self.levels)
                    
                    output_frame = detection_frame + filtered_frame
                    output_frame = cv2.convertScaleAbs(output_frame)
                    
                    # Update buffer index
                    buffer_index = (buffer_index + 1) % self.buffer_size
                    
                    # Display processed face
                    output_frame_show = cv2.resize(output_frame, (self.video_width // 2, self.video_height // 2))
                    h, w = output_frame_show.shape[:2]
                    display_frame[10:10+h, self.width-10-w:self.width-10] = output_frame_show
                    
                    # Calculate and display BPM
                    if i > self.bpm_buffer_size:
                        # Get average BPM
                        nonzero_indices = self.bpm_buffer > 0
                        if np.any(nonzero_indices):
                            avg_bpm = self.bpm_buffer[nonzero_indices].mean()
                            self.current_bpm = avg_bpm
                            self.bpm_ready = True
                            
                            # Set color based on BPM
                            if 60 <= avg_bpm <= 100:  # Normal
                                bpm_color = (0, 255, 0)
                            elif avg_bpm < 60:  # Low
                                bpm_color = (255, 255, 0)
                            else:  # High
                                bpm_color = (0, 0, 255)
                            
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
                
                # Convert to RGB for Streamlit
                display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                
                # Put frame in queue for display
                if not self.frame_queue.full():
                    self.frame_queue.put(display_frame)
                else:
                    # Get rid of the oldest frame
                    _ = self.frame_queue.get()
                    self.frame_queue.put(display_frame)
        
        except Exception as e:
            st.error(f"Error in frame processing: {str(e)}")
        finally:
            webcam.release()
            self.is_monitoring = False
    
    def get_latest_frame(self):
        if not self.frame_queue.empty():
            return self.frame_queue.get()
        return None
    
    def get_current_bpm(self):
        if self.bpm_ready:
            return int(self.current_bpm)
        return 0
    
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
        
        # Ensure dimensions match
        filtered_frame = filtered_frame[:self.video_height, :self.video_width]
        return filtered_frame

# Initialize session state for the heart rate monitor
if 'monitor' not in st.session_state:
    st.session_state.monitor = HeartRateMonitor()
if 'monitoring_active' not in st.session_state:
    st.session_state.monitoring_active = False
if 'bpm_history' not in st.session_state:
    st.session_state.bpm_history = []
if 'bpm_times' not in st.session_state:
    st.session_state.bpm_times = []

# Application layout
st.markdown('<div class="main-header">❤️ Heart Rate Monitor</div>', unsafe_allow_html=True)
st.markdown("Monitor your heart rate in real-time using computer vision technology")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Video feed
    video_placeholder = st.empty()
    
    # Control buttons
    button_col1, button_col2 = st.columns(2)
    with button_col1:
        if not st.session_state.monitoring_active:
            if st.button("▶️ Start Monitoring", use_container_width=True):
                st.session_state.monitoring_active = True
                st.session_state.monitor.start_monitoring()
                st.session_state.bpm_history = []
                st.session_state.bpm_times = []
                st.experimental_rerun()
        else:
            if st.button("⏹ Stop Monitoring", use_container_width=True):
                st.session_state.monitoring_active = False
                st.session_state.monitor.stop_monitoring()
                st.experimental_rerun()
    
    with button_col2:
        # Reset button
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.monitoring_active = False
            st.session_state.monitor.stop_monitoring()
            st.session_state.monitor = HeartRateMonitor()
            st.session_state.bpm_history = []
            st.session_state.bpm_times = []
            st.experimental_rerun()

with col2:
    # BPM display
    st.markdown('<div class="sub-header">Current Heart Rate</div>', unsafe_allow_html=True)
    
    bpm_container = st.container()
    
    # Heart rate categories
    st.markdown('<div class="sub-header">Heart Rate Categories</div>', unsafe_allow_html=True)
    
    categories_container = st.container()
    with categories_container:
        st.markdown("""
        <div class="info-box">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span>Bradycardia (Low)</span>
                <span style="background-color: #ffaa00; color: #fff; padding: 0.2rem 0.5rem; border-radius: 0.25rem;">Below 60 BPM</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span>Normal</span>
                <span style="background-color: #38b000; color: #fff; padding: 0.2rem 0.5rem; border-radius: 0.25rem;">60-100 BPM</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span>Tachycardia (High)</span>
                <span style="background-color: #d90429; color: #fff; padding: 0.2rem 0.5rem; border-radius: 0.25rem;">Above 100 BPM</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # How it works
    st.markdown('<div class="sub-header">How it works</div>', unsafe_allow_html=True)
    
    with st.expander("Read more about this technology", expanded=False):
        st.markdown("""
        This heart rate monitor uses **Eulerian Video Magnification** to detect subtle color changes in your face that correspond to blood flow with each heartbeat.
        
        ### The process:
        1. **Face Detection**: Identifies your face in the video feed
        2. **Color Analysis**: Tracks subtle color variations in facial skin
        3. **Signal Processing**: Applies filters to isolate heart rate signal
        4. **Frequency Analysis**: Calculates heart rate from the dominant frequency
        
        ### For best results:
        - Ensure good, even lighting on your face
        - Position your face clearly in the camera view
        - Stay relatively still during measurement
        - Allow 10-15 seconds for calibration
        
        ### Limitations:
        - Accuracy varies based on lighting and camera quality
        - Not a medical device - for informational purposes only
        - May not work well in poor lighting conditions
        """)

# Create a placeholder for the BPM value and chart
chart_placeholder = st.empty()

# Function to update BPM display
def update_bpm_display(bpm):
    if bpm > 0:
        # Determine status color
        if bpm < 60:
            status_class = "background-color: #ffaa00; color: white;"
            status_text = "Low"
        elif bpm <= 100:
            status_class = "background-color: #38b000; color: white;"
            status_text = "Normal"
        else:
            status_class = "background-color: #d90429; color: white;"
            status_text = "High"
        
        # Update display
        bpm_container.markdown(f"""
        <div style="text-align: center;">
            <div class="bpm-display">{bpm}</div>
            <div class="bpm-label">Beats Per Minute</div>
            <div class="bpm-status" style="{status_class}">{status_text}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        bpm_container.markdown("""
        <div style="text-align: center;">
            <div class="bpm-display">--</div>
            <div class="bpm-label">Calculating...</div>
            <div class="bpm-status" style="background-color: #4cc9f0; color: white;">Initializing</div>
        </div>
        """, unsafe_allow_html=True)

# Frame update logic
if st.session_state.monitoring_active:
    # Display frame placeholder
    frame = st.session_state.monitor.get_latest_frame()
    if frame is not None:
        video_placeholder.image(frame, channels="RGB", use_column_width=True)
    else:
        video_placeholder.markdown("""
        <div style="display: flex; justify-content: center; align-items: center; height: 480px; 
                    background-color: #f8f9fa; border-radius: 0.5rem; border: 1px solid #dee2e6;">
            <div style="text-align: center;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">⏳</div>
                <h3>Starting camera...</h3>
                <p style="color: #6c757d;">Please wait</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Get current BPM value
    bpm = st.session_state.monitor.get_current_bpm()
    update_bpm_display(bpm)
    
    # Update BPM history
    if bpm > 0:
        st.session_state.bpm_history.append(bpm)
        st.session_state.bpm_times.append(time.time())
        
        # Keep last 60 seconds of data
        if len(st.session_state.bpm_history) > 60:
            st.session_state.bpm_history.pop(0)
            st.session_state.bpm_times.pop(0)
    
    # Draw chart if we have data
    if len(st.session_state.bpm_history) > 1:
        # Convert timestamps to relative seconds
        base_time = st.session_state.bpm_times[0]
        relative_times = [t - base_time for t in st.session_state.bpm_times]
        
        # Create and display chart
        chart_data = {
            'time': relative_times,
            'bpm': st.session_state.bpm_history
        }
        
        # Calculate min/max ranges for a stable chart
        min_bpm = max(40, min(st.session_state.bpm_history) - 10)
        max_bpm = min(180, max(st.session_state.bpm_history) + 10)
        
        chart = chart_placeholder.line_chart(
            chart_data, 
            x='time', 
            y='bpm',
            height=300
        )
    
    # Add auto-refresh to keep updating
    st.experimental_rerun()
else:
    # Display placeholder when not monitoring
    video_placeholder.markdown("""
    <div style="display: flex; justify-content: center; align-items: center; height: 480px; 
                background-color: #f8f9fa; border-radius: 0.5rem; border: 1px solid #dee2e6;">
        <div style="text-align: center;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">📹</div>
            <h3>Camera feed will appear here</h3>
            <p style="color: #6c757d;">Click "Start Monitoring" to begin</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    bpm_container.markdown("""
    <div style="text-align: center;">
        <div class="bpm-display">--</div>
        <div class="bpm-label">Beats Per Minute</div>
        <div class="bpm-status" style="background-color: #6c757d; color: white;">Inactive</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Show previous BPM data if available
    if len(st.session_state.bpm_history) > 1:
        st.markdown('<div class="sub-header">Previous Session Data</div>', unsafe_allow_html=True)
        
        # Convert timestamps to relative seconds
        base_time = st.session_state.bpm_times[0]
        relative_times = [t - base_time for t in st.session_state.bpm_times]
        
        # Create and display chart
        chart_data = {
            'time': relative_times,
            'bpm': st.session_state.bpm_history
        }
        
        chart = chart_placeholder.line_chart(
            chart_data, 
            x='time', 
            y='bpm',
            height=300
        )
        
        avg_bpm = sum(st.session_state.bpm_history) / len(st.session_state.bpm_history)
        st.markdown(f"**Average BPM:** {int(avg_bpm)}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; font-size: 0.8rem;">
    <p>❗ <strong>Disclaimer:</strong> This application is for informational purposes only and not intended for medical use.</p>
    <p>Always consult healthcare professionals for medical advice.</p>
</div>
""", unsafe_allow_html=True)