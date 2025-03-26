# HealthAssist AI

A comprehensive health assessment application that combines symptom analysis and heart rate monitoring using computer vision.

## Overview

HealthAssist AI is a web-based application that offers two main features:
1. **Symptom Checker**: Analyzes user-reported symptoms to identify possible conditions
2. **Heart Rate Monitor**: Uses computer vision to detect facial features and calculate heart rate (BPM) in real-time
3. **Combined Assessment**: Integrates both analyses for a more complete health evaluation

## Features

### Symptom Checker
- Input symptoms via text interface
- Advanced natural language processing to identify medical conditions
- Probability-based condition matching
- Personalized recommendations based on identified conditions
- Medical disclaimer to ensure proper medical advice is sought

### Heart Rate Monitor
- Real-time BPM calculation using Eulerian Video Magnification
- Facial detection using OpenCV and dlib
- Live webcam feed with BPM overlay
- Customizable BPM visualization graph
- Historical BPM data tracking

### Combined Assessment
- Integrated health evaluation combining both symptom and heart rate data
- Comprehensive health status report
- Visual presentation of critical health information
- Triage recommendations based on combined analysis

## Technology Stack

- **Backend**: Python, Flask
- **Computer Vision**: OpenCV, dlib, NumPy, SciPy
- **Frontend**: JavaScript, HTML5, CSS3, Bootstrap 5
- **Data Visualization**: Chart.js
- **Medical Data**: Custom medical knowledge base

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Webcam access for BPM monitoring

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/healthassist-ai.git
   cd healthassist-ai
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv .venv
   # On Windows
   .venv\Scripts\activate
   # On macOS/Linux
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Download the shape predictor model (for face detection):
   ```
   # Download from dlib's website or use a package manager
   # Place the file in the project root directory
   # File name: shape_predictor_68_face_landmarks.dat
   ```

5. Run the application:
   ```
   python run.py
   ```

6. Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

## Usage

### Symptom Checker
1. Navigate to the "Symptom Checker" page
2. Enter symptoms separated by commas (e.g., "headache, fever, cough")
3. Click "Analyze Symptoms" to receive an assessment

### Heart Rate Monitor
1. Navigate to the "Heart Rate Monitor" page
2. Ensure your face is visible to the webcam
3. Click "Start Monitoring" to begin BPM calculation
4. Remain still for accurate readings
5. Customize the graph display using the provided controls

### Combined Assessment
1. Navigate to the "Complete Assessment" page
2. Enter symptoms and start heart rate monitoring
3. View the comprehensive health assessment that combines both analyses

## Customizing BPM Graph

The BPM visualization graph can be customized with the following options:

- **Time Range**: Choose between 30 seconds, 1 minute, 2 minutes, or 5 minutes of data
- **Line Color**: Select any color for the graph line
- **Line Thickness**: Choose between thin, normal, or thick line styles

## Privacy Notice

This application processes health data and webcam footage locally in your browser. No data is sent to external servers except for API calls required for symptom analysis. Webcam access is only used for BPM calculation and is never recorded or stored.

## Disclaimer

HealthAssist AI is for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenCV and dlib for computer vision capabilities
- Chart.js for data visualization
- Bootstrap for UI components
- All the contributors to the open-source libraries used in this project 