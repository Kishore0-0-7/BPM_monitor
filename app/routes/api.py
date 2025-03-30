from flask import Blueprint, request, jsonify, current_app
from app.routes.main import symptom_analyzer, bpm_monitor
import uuid
import json
import os
import time

api_bp = Blueprint('api', __name__)

@api_bp.route('/analyze-symptoms', methods=['POST'])
def analyze_symptoms():
    try:
        data = request.get_json()
        symptom_input = data.get('symptoms', [])
        user_info = data.get('user_info', {})
        
        # Handle different input formats
        if isinstance(symptom_input, str):
            symptoms = [symptom_input]
        elif isinstance(symptom_input, list):
            symptoms = symptom_input
        else:
            current_app.logger.error(f"Invalid symptom format: {type(symptom_input)}")
            return jsonify({'error': 'Symptoms must be a string or list of strings'}), 400
            
        # Clean up symptoms - remove empty entries and strip whitespace
        symptoms = [s.strip() for s in symptoms if s and s.strip()]
        current_app.logger.info(f"Processing symptoms: {symptoms}")
        
        if not symptoms:
            current_app.logger.warning("No valid symptoms provided for analysis")
            return jsonify({
                'possible_conditions': [],
                'recommendations': ["Please enter symptoms for analysis"],
                'identified_symptoms': [],
                'unidentified_symptoms': []
            })
        
        # Analyze symptoms
        current_app.logger.info(f"Starting analysis of symptoms: {symptoms}")
        results = symptom_analyzer.analyze(symptoms)
        current_app.logger.info(f"Analysis completed with results: {results}")
        
        # Add unique analysis ID
        results['analysis_id'] = str(uuid.uuid4())
        
        # Store analysis for later reference
        _store_analysis(results)
        
        return jsonify(results)
    except Exception as e:
        current_app.logger.error(f"Error analyzing symptoms: {str(e)}", exc_info=True)
        return jsonify({'error': f'An error occurred during analysis: {str(e)}'}), 500

@api_bp.route('/start-bpm-monitor', methods=['POST'])
def start_bpm_monitor():
    try:
        success = bpm_monitor.start_monitoring()
        return jsonify({'success': success})
    except Exception as e:
        current_app.logger.error(f"Error starting BPM monitor: {str(e)}")
        return jsonify({'error': 'Could not start BPM monitoring'}), 500

@api_bp.route('/stop-bpm-monitor', methods=['POST'])
def stop_bpm_monitor():
    try:
        success = bpm_monitor.stop_monitoring()
        return jsonify({'success': success})
    except Exception as e:
        current_app.logger.error(f"Error stopping BPM monitor: {str(e)}")
        return jsonify({'error': 'Could not stop BPM monitoring'}), 500

@api_bp.route('/get-bpm', methods=['GET'])
def get_bpm():
    try:
        bpm_data = bpm_monitor.get_current_bpm()
        return jsonify(bpm_data)
    except Exception as e:
        current_app.logger.error(f"Error getting BPM: {str(e)}")
        return jsonify({'error': 'Could not get BPM data'}), 500

@api_bp.route('/get-frame', methods=['GET'])
def get_frame():
    try:
        frame = bpm_monitor.get_latest_frame()
        if frame:
            return jsonify({'frame': frame})
        else:
            return jsonify({'frame': None})
    except Exception as e:
        current_app.logger.error(f"Error getting frame: {str(e)}")
        return jsonify({'error': 'Could not get video frame'}), 500

@api_bp.route('/combined-health-check', methods=['POST'])
def combined_health_check():
    try:
        data = request.get_json()
        symptoms = data.get('symptoms', [])
        bpm = data.get('bpm', 0)
        user_info = data.get('user_info', {})
        
        # Analyze symptoms
        symptom_results = symptom_analyzer.analyze(symptoms)
        
        # Add BPM analysis
        bpm_analysis = _analyze_bpm(bpm)
        
        # Combine results
        combined_results = {
            'symptom_analysis': symptom_results,
            'bpm_analysis': bpm_analysis,
            'combined_assessment': _generate_combined_assessment(symptom_results, bpm_analysis),
            'analysis_id': str(uuid.uuid4())
        }
        
        # Store analysis
        _store_analysis(combined_results)
        
        return jsonify(combined_results)
    except Exception as e:
        current_app.logger.error(f"Error in combined health check: {str(e)}")
        return jsonify({'error': 'An error occurred during analysis'}), 500

def _analyze_bpm(bpm):
    """Analyze BPM value and provide assessment"""
    if not bpm or bpm == 0:
        return {
            'status': 'unknown',
            'message': 'BPM not available',
            'recommendation': 'Please use the BPM monitor to measure your heart rate.'
        }
    
    # Analyze BPM
    if bpm < 60:
        return {
            'status': 'low',
            'message': 'Your heart rate is below the normal range (60-100 BPM).',
            'bpm': bpm,
            'recommendation': 'Low heart rate may indicate excellent cardiovascular fitness in athletes, but can also be a sign of certain medical conditions. If you experience symptoms like dizziness or fatigue, consult a healthcare provider.'
        }
    elif 60 <= bpm <= 100:
        return {
            'status': 'normal',
            'message': 'Your heart rate is within the normal range (60-100 BPM).',
            'bpm': bpm,
            'recommendation': 'Your heart rate appears normal. Continue to maintain a healthy lifestyle with regular exercise and balanced diet.'
        }
    else:
        return {
            'status': 'high',
            'message': 'Your heart rate is above the normal range (60-100 BPM).',
            'bpm': bpm,
            'recommendation': 'Elevated heart rate can be due to stress, caffeine, medication, or physical activity. If persistently high at rest, consult a healthcare provider.'
        }

def _generate_combined_assessment(symptom_results, bpm_analysis):
    """Generate combined assessment from symptoms and BPM"""
    if not symptom_results.get('possible_conditions'):
        return {
            'conclusion': 'Insufficient data for comprehensive assessment',
            'recommendation': 'Please provide more symptom information for a better assessment.'
        }
    
    top_condition = symptom_results['possible_conditions'][0]
    severity = top_condition.get('severity', 1)
    bpm_status = bpm_analysis.get('status', 'unknown')
    
    # Assess combined risk
    if severity >= 3 and bpm_status != 'normal':
        risk_level = 'high'
    elif severity >= 2 or bpm_status != 'normal':
        risk_level = 'moderate'
    else:
        risk_level = 'low'
    
    # Generate combined recommendations
    combined_recommendations = []
    
    # Add top condition-specific recommendations
    if top_condition['name'] != 'Unspecified Condition':
        combined_recommendations.append(
            f"Based on your symptoms, you may have {top_condition['name']}."
        )
    
    # Add symptom-based recommendations
    combined_recommendations.extend(symptom_results.get('recommendations', []))
    
    # Add BPM-specific recommendations if available
    if bpm_status != 'unknown':
        combined_recommendations.append(bpm_analysis.get('recommendation', ''))
    
    return {
        'risk_level': risk_level,
        'conclusion': _generate_conclusion(top_condition, bpm_analysis, risk_level),
        'recommendations': combined_recommendations
    }

def _generate_conclusion(top_condition, bpm_analysis, risk_level):
    """Generate a conclusion based on the analysis"""
    condition_name = top_condition.get('name', 'Unspecified Condition')
    bpm_status = bpm_analysis.get('status', 'unknown')
    
    if condition_name == 'Unspecified Condition':
        if bpm_status == 'unknown':
            return "Based on the limited information provided, we cannot make a specific assessment."
        else:
            return f"Your symptoms don't clearly match a specific condition, but your heart rate is {bpm_status}."
    
    if bpm_status == 'unknown':
        return f"Your symptoms suggest {condition_name}. Consider measuring your heart rate for a more complete assessment."
    
    if risk_level == 'high':
        return f"Your symptoms indicate possible {condition_name} and your heart rate is {bpm_status}. Please seek medical attention promptly."
    elif risk_level == 'moderate':
        return f"Your symptoms suggest {condition_name} and your heart rate is {bpm_status}. Monitor your condition closely."
    else:
        return f"Your symptoms may indicate {condition_name} and your heart rate is {bpm_status}. Your overall risk appears low."

def _store_analysis(analysis_data):
    """Store analysis results for later reference"""
    try:
        # Create directory if it doesn't exist
        analysis_dir = os.path.join(current_app.config.get('MEDICAL_DATA_PATH'), 'analysis')
        if not os.path.exists(analysis_dir):
            os.makedirs(analysis_dir)
        
        # Generate filename with timestamp and analysis ID
        analysis_id = analysis_data.get('analysis_id', str(uuid.uuid4()))
        timestamp = int(time.time())
        filename = f"{timestamp}_{analysis_id}.json"
        
        # Write analysis to file
        with open(os.path.join(analysis_dir, filename), 'w') as f:
            json.dump(analysis_data, f, indent=2)
    except Exception as e:
        current_app.logger.error(f"Error storing analysis: {str(e)}") 