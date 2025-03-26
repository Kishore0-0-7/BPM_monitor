# HealthAssist AI - Technical Documentation

This document outlines the development process, implementation details, and technical considerations for the HealthAssist AI project.

## Project Structure

```
healthassist-ai/
├── app/
│   ├── __init__.py         # Flask application factory
│   ├── models/
│   │   ├── __init__.py
│   │   ├── bpm_monitor.py  # Heart rate monitoring implementation
│   │   └── symptom_analyzer.py  # Symptom analysis implementation
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── api.py          # API endpoints
│   │   └── main.py         # Main routes for web pages
│   ├── static/
│   │   ├── css/            # Stylesheets
│   │   ├── js/             # Client-side JavaScript
│   │   └── img/            # Static images
│   └── templates/          # Jinja2 templates
│       ├── base.html       # Base template with common layout
│       ├── index.html      # Homepage
│       ├── symptom-checker.html   # Symptom analysis page
│       ├── bpm-monitor.html       # Heart rate monitoring page
│       └── combined-assessment.html  # Integrated assessment page
├── config.py               # Application configuration
├── run.py                  # Application entry point
├── requirements.txt        # Python dependencies
├── README.md               # User documentation
└── WORKING.md              # Technical documentation (this file)
```

## Core Components

### BPM Monitor (Heart Rate Detection)

The `BPMMonitor` class in `app/models/bpm_monitor.py` implements real-time heart rate monitoring using computer vision techniques.

#### Technical Implementation

1. **Face Detection**:
   - Uses dlib's frontal face detector to locate the face in each video frame
   - Optional landmarks detection for improved region of interest (ROI) selection

2. **Eulerian Video Magnification (EVM)**:
   - Applies Gaussian pyramid decomposition to isolate subtle color changes in skin
   - Uses temporal filtering to amplify blood volume pulse signal
   - Implemented based on the MIT CSAIL paper "Eulerian Video Magnification for Revealing Subtle Changes in Video"

3. **Signal Processing**:
   - Applies bandpass filtering to isolate frequencies in the 0.83-3.0 Hz range (50-180 BPM)
   - Uses Fast Fourier Transform (FFT) to extract frequency components
   - Calculates BPM from the dominant frequency

4. **Multithreaded Processing**:
   - Separate threads for video capture and signal processing
   - Thread-safe queues for frame exchange
   - Background processing to maintain UI responsiveness

#### Key Methods

- `start_monitoring()`: Initializes threads and begins the monitoring process
- `stop_monitoring()`: Safely terminates monitoring threads
- `get_current_bpm()`: Returns the current calculated BPM value
- `get_latest_frame()`: Returns the latest processed frame with visualizations
- `_build_gauss()`: Creates a Gaussian pyramid for EVM
- `_reconstruct_frame()`: Reconstructs the processed frame from the pyramid
- `_process_frames()`: Main signal processing pipeline

### Symptom Analyzer

The `SymptomAnalyzer` class in `app/models/symptom_analyzer.py` processes user-provided symptoms to identify possible medical conditions.

#### Technical Implementation

1. **Symptom Matching**:
   - Implements fuzzy matching of user input to standardized medical terminology
   - Uses string similarity metrics to handle variations in symptom descriptions

2. **Condition Probability Calculation**:
   - Calculates relevance scores for each condition based on matched symptoms
   - Considers symptom specificity and condition prevalence
   - Uses Bayesian probability to estimate likelihood of conditions

3. **Recommendation Generation**:
   - Creates personalized recommendations based on identified conditions
   - Implements triage logic to suggest appropriate level of care

#### Medical Data Structure

```json
{
  "symptoms": {
    "symptom_id": {
      "name": "Symptom name",
      "description": "Detailed description",
      "aliases": ["alternative names", "common terms"]
    }
  },
  "conditions": {
    "condition_id": {
      "name": "Condition name",
      "description": "Detailed description",
      "symptoms": ["symptom_id1", "symptom_id2", ...],
      "severity": 1-5,
      "triage_level": "Self-care|Primary care|Urgent care|Emergency",
      "medical_specialty": "Relevant specialty"
    }
  }
}
```

### Web Interface

The web interface is built using Flask, Bootstrap 5, and JavaScript.

#### Flask Routes

- Main pages (`routes/main.py`):
  - `/`: Homepage
  - `/symptom-checker`: Symptom analysis page
  - `/bpm-monitor`: Heart rate monitoring page
  - `/combined-assessment`: Integrated assessment page

- API endpoints (`routes/api.py`):
  - `/api/analyze-symptoms`: Process symptoms and return analysis
  - `/api/start-bpm-monitor`: Start heart rate monitoring
  - `/api/stop-bpm-monitor`: Stop heart rate monitoring
  - `/api/get-bpm`: Get current BPM reading
  - `/api/get-frame`: Get latest webcam frame with visualization
  - `/api/combined-health-check`: Perform integrated health assessment

#### Client-Side Implementation

1. **BPM Visualization**:
   - Real-time chart using Chart.js
   - Customizable appearance (time range, colors, thickness)
   - Historical data display

2. **Webcam Integration**:
   - Secure webcam access using browser APIs
   - Base64 encoding for frame transmission
   - Real-time display of processed frames

3. **Responsive Design**:
   - Mobile-friendly interface using Bootstrap 5
   - Adaptive layout for different screen sizes
   - Accessible UI elements

## Development Notes

### Performance Considerations

1. **BPM Calculation Optimization**:
   - Downscaling video frames to 320x240 for faster processing
   - Using a fixed buffer size for frequency analysis
   - Caching intermediate results to avoid redundant calculations

2. **API Response Optimization**:
   - Limiting frame rate for webcam image transmission
   - Using efficient JSON serialization
   - Implementing polling intervals to reduce server load

### Security Considerations

1. **Privacy Protection**:
   - Webcam processing done entirely client-side
   - No persistent storage of health data or video frames
   - No third-party analytics or tracking

2. **Input Validation**:
   - Sanitizing all user inputs
   - Validating API parameters
   - Protection against common web vulnerabilities

### Known Limitations

1. **BPM Monitoring**:
   - Requires good lighting conditions
   - Sensitive to movement
   - Works best with front-facing, unobstructed view of the face
   - May be less accurate for darker skin tones (an area for improvement)

2. **Symptom Analysis**:
   - Limited medical knowledge base
   - Cannot capture nuanced symptom descriptions
   - Not a replacement for professional diagnosis

## Future Enhancements

1. **Technical Improvements**:
   - Implement machine learning for improved facial blood flow detection
   - Add support for multiple face tracking
   - Optimize for mobile devices
   - Implement progressive web app (PWA) capabilities

2. **Feature Enhancements**:
   - User accounts and history tracking
   - Integration with wearable devices
   - Multi-language support
   - Accessibility improvements
   - Offline functionality

3. **Medical Capabilities**:
   - Expanded medical knowledge base
   - Integration with clinical guidelines
   - Support for chronic condition monitoring
   - Medication tracking and reminders

## Development Setup Instructions

1. **Environment Setup**:
   ```bash
   # Create virtual environment
   python -m venv .venv
   
   # Activate virtual environment
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Development Server**:
   ```bash
   # Run with debug mode
   python run.py
   ```

3. **Testing**:
   ```bash
   # Run tests
   pytest
   
   # Run tests with coverage report
   pytest --cov=app
   ```

## Deployment Considerations

1. **Production Server**:
   - Use a production WSGI server (Gunicorn, uWSGI)
   - Set `debug=False` in production
   - Implement proper error logging

2. **Environment Variables**:
   - Store sensitive configuration in environment variables
   - Use different configurations for development and production

3. **SSL/TLS**:
   - Always use HTTPS in production
   - Required for secure webcam access
   - Set up proper SSL certificates

4. **Resource Requirements**:
   - Minimum 1GB RAM for server
   - CPU with AVX2 instructions recommended for optimal dlib performance
   - SSD storage for faster loading of models and data 