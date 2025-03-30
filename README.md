# HealthAssist AI - Heart Rate Monitor & Symptom Analyzer

HealthAssist AI is a comprehensive health monitoring system that combines real-time heart rate detection with symptom analysis. The application uses computer vision techniques to detect heart rate from facial video feed and provides medical symptom analysis.

## Features

### Heart Rate Monitoring
- Non-contact heart rate detection using webcam
- Eulerian Video Magnification to amplify subtle color changes in skin
- Real-time pulse waveform visualization
- BPM (Beats Per Minute) tracking and logging
- Heart rate zone categorization

### Symptom Analysis
- Symptom matching against medical database
- Medical condition probability assessment
- Personalized recommendations based on symptoms
- Severity classification

### Reporting
- Combined health reports with BPM data and symptom analysis
- Historical session tracking
- PDF and JSON export options

## Installation

### Prerequisites
- Python 3.8 or higher
- Webcam
- (Optional) NVIDIA GPU for improved performance

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd BPM
```

2. Create a virtual environment and activate it:
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On MacOS/Linux:
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the facial landmark predictor model:
```bash
# Option 1: Manual download (recommended)
# Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
# Extract the .bz2 file and place shape_predictor_68_face_landmarks.dat in the project root

# Option 2: Using Python (if bz2 module is available)
python -c "import bz2; import requests; open('shape_predictor_68_face_landmarks.dat', 'wb').write(bz2.decompress(requests.get('http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2').content))"
```

5. Create necessary data directories:
```bash
python create_data_dir.py
```

## Usage

### Running the Flask Web Application

To run the Flask-based heart rate monitor:

```bash
python bpm_app.py
```

Then open your browser and navigate to:
```
http://127.0.0.1:5000/
```

### Running the Streamlit Dashboard

To run the comprehensive Streamlit dashboard with all features:

```bash
streamlit run streamlit_app.py
```

This will automatically open the application in your default browser.

## Technical Details

### Heart Rate Detection Technique

The application uses Eulerian Video Magnification (EVM) to detect heart rate:

1. **Face Detection**: Using dlib's face detector and landmark predictor to locate the face and important facial points.
2. **Region of Interest (ROI)**: The forehead region is extracted as it has good blood perfusion and less movement.
3. **Color Magnification**: Subtle color changes in the skin due to blood flow are amplified.
4. **Frequency Analysis**: Fast Fourier Transform (FFT) is used to extract the dominant frequency (heart rate).
5. **Signal Processing**: Various filters are applied to isolate the heart rate signal from noise.

### Optimization Tips

For better performance and accuracy:

- Ensure good, consistent lighting on your face
- Minimize movement during monitoring
- Keep your face centered and at a reasonable distance from the camera
- If using a laptop, connect to power for more consistent webcam performance
- A higher quality webcam will generally yield better results

## Troubleshooting

### Common Issues

#### Face Detection Problems
- **Issue**: Application cannot detect a face
- **Solution**: Improve lighting, ensure face is clearly visible, and try different angles

#### Heart Rate Accuracy
- **Issue**: Heart rate readings seem inaccurate
- **Solution**: Stay still, improve lighting, ensure the forehead is clearly visible

#### Performance Issues
- **Issue**: Application runs slowly
- **Solution**: Close other applications using the webcam, reduce resolution in settings

### Dependency Issues

If you encounter problems with dlib installation:
- On Windows, try using the pre-built wheel: `pip install dlib-19.22.99-cp39-cp39-win_amd64.whl` (adjust for your Python version)
- On Linux, ensure you have CMake and C++ build tools installed: `sudo apt-get install cmake build-essential`

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MIT CSAIL for the Eulerian Video Magnification research
- dlib creators for the face detection algorithms

## About

This project was developed as a showcase of advanced computer vision techniques applied to health monitoring. It is intended for educational and research purposes and is not a certified medical device. 