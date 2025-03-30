from flask import Blueprint, render_template, request, jsonify, current_app
from app.models.symptom_analyzer import SymptomAnalyzer
from app.models.bpm_monitor import BPMMonitor
import os
from config import Config

main_bp = Blueprint('main', __name__)

# Initialize models directly (simple but may not be ideal for all scenarios)
try:
    data_path = Config.MEDICAL_DATA_PATH
    symptom_analyzer = SymptomAnalyzer(data_path)
    bpm_monitor = BPMMonitor()
except Exception as e:
    print(f"Error initializing models: {e}")
    symptom_analyzer = None
    bpm_monitor = None

@main_bp.route('/')
def index():
    return render_template('index.html')

@main_bp.route('/symptom-checker')
def symptom_checker():
    return render_template('symptom-checker.html')

@main_bp.route('/bpm-monitor')
def bpm_monitor_page():
    return render_template('bpm-monitor.html')

@main_bp.route('/combined-assessment')
def combined_assessment():
    return render_template('combined-assessment.html') 