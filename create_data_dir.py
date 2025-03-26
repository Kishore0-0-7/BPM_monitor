import os
import json

# Create data directory if it doesn't exist
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Create default symptom data
symptoms = {
    "headache": {"id": "S1", "name": "Headache", "description": "Pain in the head or upper neck"},
    "fever": {"id": "S2", "name": "Fever", "description": "Elevated body temperature above normal range"},
    "cough": {"id": "S3", "name": "Cough", "description": "Sudden expulsion of air from the lungs"},
    "fatigue": {"id": "S4", "name": "Fatigue", "description": "Extreme tiredness or exhaustion"},
    "nausea": {"id": "S5", "name": "Nausea", "description": "Sensation of unease in the stomach with an urge to vomit"}
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
    }
}

# Create default mapping data
mappings = {
    "common_cold": ["headache", "cough", "fatigue"],
    "influenza": ["fever", "headache", "fatigue", "cough"]
}

# Write files
with open(os.path.join(data_dir, 'symptoms.json'), 'w') as f:
    json.dump(symptoms, f, indent=2)

with open(os.path.join(data_dir, 'conditions.json'), 'w') as f:
    json.dump(conditions, f, indent=2)

with open(os.path.join(data_dir, 'symptom_condition_mapping.json'), 'w') as f:
    json.dump(mappings, f, indent=2)

print("Data directory and files created successfully!") 