from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from simple_symptom_analyzer import SimpleSymptomAnalyzer as SymptomAnalyzer
import os
import traceback
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Initialize our symptom analyzer model
try:
    print("Initializing SymptomAnalyzer...")
    symptom_analyzer = SymptomAnalyzer()
    print("SymptomAnalyzer initialized successfully!")
except Exception as e:
    print(f"Error initializing SymptomAnalyzer: {str(e)}")
    print(traceback.format_exc())
    symptom_analyzer = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_symptoms():
    # Get data from request
    data = request.get_json()
    symptoms = data.get('symptoms', [])
    user_info = data.get('user_info', {})
    
    print(f"Received symptoms: {symptoms}")
    
    # Validate input
    if not symptoms:
        return jsonify({'error': 'No symptoms provided'}), 400
    
    # Analyze symptoms
    try:
        if symptom_analyzer is None:
            return jsonify({'error': 'Symptom analyzer not initialized'}), 500
            
        analysis = symptom_analyzer.analyze(symptoms, user_info)
        print(f"Analysis result: {analysis}")
        return jsonify(analysis)
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    # Get feedback data
    data = request.get_json()
    diagnosis_id = data.get('diagnosis_id')
    accurate = data.get('accurate')
    actual_condition = data.get('actual_condition', '')
    
    # Store feedback to improve the model
    try:
        symptom_analyzer.store_feedback(diagnosis_id, accurate, actual_condition)
        return jsonify({'status': 'success'})
    except Exception as e:
        print(f"Error saving feedback: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask application...")
    app.run(debug=True, port=5000) 