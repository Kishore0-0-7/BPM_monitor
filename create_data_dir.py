import os
import json

# Create data directories if they don't exist
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dirs = [
    os.path.join(base_dir, 'data'),
    os.path.join(base_dir, 'data', 'bpm_sessions'),
    os.path.join(base_dir, 'data', 'medical_data'),
    os.path.join(base_dir, 'data', 'reports'),
    os.path.join(base_dir, 'data', 'analysis'),
    os.path.join(base_dir, 'templates')
]

for dir_path in data_dirs:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Created directory: {dir_path}")

# Create default template file if it doesn't exist
template_file = os.path.join(base_dir, 'templates', 'index.html')
if not os.path.exists(template_file):
    with open(template_file, 'w') as f:
        f.write("""<!DOCTYPE html>
<html>
<head>
    <title>Heart Rate Monitor - HealthAssist AI</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background-color: #4a69bd;
            color: white;
            padding: 15px;
            text-align: center;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .container {
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .video-container {
            width: 100%;
            max-width: 640px;
            margin-bottom: 20px;
            border: 1px solid #ddd;
            border-radius: 5px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .video-feed {
            width: 100%;
            height: auto;
        }
        .bpm-display {
            background-color: white;
            border-radius: 5px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            width: 100%;
            max-width: 640px;
        }
        .bpm-value {
            font-size: 48px;
            font-weight: bold;
            margin: 10px 0;
        }
        .normal { color: #2ecc71; }
        .low { color: #3498db; }
        .elevated { color: #e67e22; }
        .high { color: #e74c3c; }
        .controls {
            display: flex;
            gap: 10px;
            margin-top: 20px;
        }
        .btn {
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
            transition: background-color 0.3s;
        }
        .btn-start {
            background-color: #2ecc71;
            color: white;
        }
        .btn-stop {
            background-color: #e74c3c;
            color: white;
        }
        .btn:hover {
            opacity: 0.8;
        }
        .status {
            margin-top: 10px;
            font-style: italic;
            color: #7f8c8d;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>❤️ HealthAssist AI</h1>
        <p>Real-time Heart Rate Monitor using Computer Vision</p>
    </div>
    
    <div class="container">
        <div class="video-container">
            <img src="{{ url_for('video_feed') }}" class="video-feed" alt="Video Feed">
        </div>
        
        <div class="bpm-display">
            <h2>Current Heart Rate</h2>
            <div id="bpm-value" class="bpm-value normal">--</div>
            <div id="bpm-status" class="status">Please wait, initializing camera...</div>
            
            <div class="controls">
                <button class="btn btn-start" id="btn-start">Start Monitoring</button>
                <button class="btn btn-stop" id="btn-stop" disabled>Stop Monitoring</button>
            </div>
        </div>
    </div>

    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const bpmValue = document.getElementById('bpm-value');
            const bpmStatus = document.getElementById('bpm-status');
            const startBtn = document.getElementById('btn-start');
            const stopBtn = document.getElementById('btn-stop');
            
            // Polling interval for BPM updates
            let pollingInterval = null;
            
            // Start monitoring
            startBtn.addEventListener('click', function() {
                startBtn.disabled = true;
                stopBtn.disabled = false;
                bpmStatus.textContent = "Monitoring your heart rate...";
                
                // Start polling for BPM updates
                pollingInterval = setInterval(updateBPM, 1000);
            });
            
            // Stop monitoring
            stopBtn.addEventListener('click', function() {
                startBtn.disabled = false;
                stopBtn.disabled = true;
                bpmStatus.textContent = "Monitoring stopped";
                
                // Clear polling interval
                clearInterval(pollingInterval);
            });
            
            // Function to update BPM display
            function updateBPM() {
                fetch('/get_bpm')
                    .then(response => response.json())
                    .then(data => {
                        if (data.status === 'ready') {
                            bpmValue.textContent = data.bpm;
                            
                            // Update color based on BPM range
                            if (data.bpm < 60) {
                                bpmValue.className = 'bpm-value low';
                                bpmStatus.textContent = "Low heart rate detected";
                            } else if (data.bpm > 100) {
                                bpmValue.className = 'bpm-value elevated';
                                bpmStatus.textContent = "Elevated heart rate detected";
                            } else if (data.bpm > 140) {
                                bpmValue.className = 'bpm-value high';
                                bpmStatus.textContent = "High heart rate detected!";
                            } else {
                                bpmValue.className = 'bpm-value normal';
                                bpmStatus.textContent = "Normal heart rate";
                            }
                        } else {
                            bpmStatus.textContent = "Calculating heart rate, please wait...";
                        }
                    })
                    .catch(error => {
                        console.error('Error fetching BPM:', error);
                        bpmStatus.textContent = "Error updating heart rate";
                    });
            }
        });
    </script>
</body>
</html>""")
    print(f"Created template file: {template_file}")

# Create default symptom data
data_dir = os.path.join(base_dir, 'data')
symptoms = {
    "headache": {"id": "S1", "name": "Headache", "description": "Pain in the head or upper neck"},
    "fever": {"id": "S2", "name": "Fever", "description": "Elevated body temperature above normal range"},
    "cough": {"id": "S3", "name": "Cough", "description": "Sudden expulsion of air from the lungs"},
    "fatigue": {"id": "S4", "name": "Fatigue", "description": "Extreme tiredness or exhaustion"},
    "nausea": {"id": "S5", "name": "Nausea", "description": "Sensation of unease in the stomach with an urge to vomit"},
    "shortness_of_breath": {"id": "S6", "name": "Shortness of Breath", "description": "Difficulty breathing or dyspnea"},
    "sore_throat": {"id": "S7", "name": "Sore Throat", "description": "Pain or irritation in the throat"},
    "dizziness": {"id": "S8", "name": "Dizziness", "description": "Feeling faint, woozy, or unsteady"}
}

# Create default condition data
conditions = {
    "common_cold": {
        "id": "C1", 
        "name": "Common Cold",
        "description": "A viral infectious disease of the upper respiratory tract",
        "severity": 1,
        "triage_level": "Self-care",
        "medical_specialty": "General Practice"
    },
    "influenza": {
        "id": "C2", 
        "name": "Influenza",
        "description": "A viral infection that attacks respiratory system",
        "severity": 2,
        "triage_level": "Non-urgent medical care",
        "medical_specialty": "General Practice"
    },
    "covid": {
        "id": "C3",
        "name": "COVID-19",
        "description": "A respiratory illness caused by the SARS-CoV-2 virus",
        "severity": 3,
        "triage_level": "Urgent medical care",
        "medical_specialty": "Infectious Disease"
    },
    "migraine": {
        "id": "C4",
        "name": "Migraine",
        "description": "A primary headache disorder characterized by recurrent headaches",
        "severity": 2,
        "triage_level": "Non-urgent medical care",
        "medical_specialty": "Neurology"
    },
    "dehydration": {
        "id": "C5",
        "name": "Dehydration",
        "description": "A condition caused by the loss of too much fluid from the body",
        "severity": 2,
        "triage_level": "Non-urgent medical care",
        "medical_specialty": "General Practice"
    },
    "anxiety": {
        "id": "C6",
        "name": "Anxiety",
        "description": "A feeling of worry, nervousness, or unease about something",
        "severity": 2,
        "triage_level": "Non-urgent medical care",
        "medical_specialty": "Mental Health"
    },
    "allergies": {
        "id": "C7",
        "name": "Allergies",
        "description": "An immune system response to a substance that most people don't react to",
        "severity": 1,
        "triage_level": "Self-care",
        "medical_specialty": "Allergy and Immunology"
    }
}

# Create default mapping data
mappings = {
    "headache": ["common_cold", "influenza", "covid", "migraine", "dehydration", "anxiety"],
    "fever": ["common_cold", "influenza", "covid"],
    "cough": ["common_cold", "influenza", "covid", "allergies"],
    "fatigue": ["common_cold", "influenza", "covid", "dehydration", "anxiety"],
    "nausea": ["influenza", "migraine", "dehydration", "anxiety"],
    "shortness_of_breath": ["covid", "anxiety", "allergies"],
    "sore_throat": ["common_cold", "influenza", "allergies"],
    "dizziness": ["dehydration", "anxiety", "migraine"]
}

# Write files
with open(os.path.join(data_dir, 'symptoms.json'), 'w') as f:
    json.dump(symptoms, f, indent=2)
    print(f"Created/Updated symptoms file")

with open(os.path.join(data_dir, 'conditions.json'), 'w') as f:
    json.dump(conditions, f, indent=2)
    print(f"Created/Updated conditions file")

with open(os.path.join(data_dir, 'symptom_condition_mapping.json'), 'w') as f:
    json.dump(mappings, f, indent=2)
    print(f"Created/Updated symptom-condition mapping file")

print("Data directory and files created successfully!") 