import os
import json
import uuid
import random

class SimpleSymptomAnalyzer:
    def __init__(self):
        # Sample data mapping symptoms to conditions
        self.condition_data = {
            'headache': ['Common Cold', 'Migraine', 'Tension Headache'],
            'fever': ['Common Cold', 'Flu', 'COVID-19'],
            'cough': ['Common Cold', 'Flu', 'Bronchitis'],
            'fatigue': ['Common Cold', 'Flu', 'Anemia'],
            'sore throat': ['Common Cold', 'Strep Throat', 'Tonsillitis'],
            'runny nose': ['Common Cold', 'Allergies', 'Sinusitis'],
            'chest pain': ['Angina', 'Heart Attack', 'Pneumonia'],
            'abdominal pain': ['Gastroenteritis', 'Appendicitis', 'IBS'],
            'nausea': ['Food Poisoning', 'Migraine', 'Gastroenteritis'],
            'dizziness': ['Vertigo', 'Low Blood Pressure', 'Anemia']
        }
        
        self.severity_data = {
            'Common Cold': 1,
            'Migraine': 2,
            'Tension Headache': 1,
            'Flu': 2,
            'COVID-19': 3,
            'Bronchitis': 2,
            'Anemia': 2,
            'Strep Throat': 2,
            'Tonsillitis': 2,
            'Allergies': 1,
            'Sinusitis': 2,
            'Angina': 3,
            'Heart Attack': 4,
            'Pneumonia': 3,
            'Gastroenteritis': 2,
            'Appendicitis': 4,
            'IBS': 2,
            'Food Poisoning': 2,
            'Vertigo': 2,
            'Low Blood Pressure': 2
        }
    
    def analyze(self, symptoms, user_info=None):
        """Simple analysis based on matching symptoms"""
        # Get all possible conditions for the symptoms
        possible_conditions = []
        for symptom in symptoms:
            if symptom.lower() in self.condition_data:
                possible_conditions.extend(self.condition_data[symptom.lower()])
        
        # Count occurrences of each condition
        condition_count = {}
        for condition in possible_conditions:
            if condition in condition_count:
                condition_count[condition] += 1
            else:
                condition_count[condition] = 1
        
        # Sort by count
        sorted_conditions = sorted(condition_count.items(), 
                                   key=lambda x: x[1], 
                                   reverse=True)
        
        # Take top 3 conditions or fewer if less than 3 found
        top_conditions = sorted_conditions[:min(3, len(sorted_conditions))]
        
        # Create result with probabilities
        results = []
        total = sum([count for _, count in top_conditions]) if top_conditions else 1
        
        if not top_conditions:
            # If no conditions found, provide a generic response
            results = [{
                'condition': 'Unspecified Condition',
                'probability': 1.0,
                'severity': 1
            }]
        else:
            for condition, count in top_conditions:
                # Calculate a probability based on count
                probability = count / total
                severity = self.severity_data.get(condition, 1)
                
                results.append({
                    'condition': condition,
                    'probability': probability,
                    'severity': severity
                })
        
        # Generate diagnosis ID for feedback
        diagnosis_id = str(uuid.uuid4())
        
        # Generate recommendations based on highest probability condition
        if results:
            recommendations = self.generate_recommendations(
                results[0]['condition'], 
                results[0]['severity']
            )
        else:
            recommendations = [
                "Insufficient symptoms to provide specific recommendations",
                "Consider consulting a healthcare provider if symptoms persist"
            ]
        
        return {
            'diagnosis_id': diagnosis_id,
            'possible_conditions': results,
            'recommendations': recommendations,
            'disclaimer': 'This is not a substitute for professional medical advice. Please consult a healthcare provider for proper diagnosis and treatment.'
        }
    
    def generate_recommendations(self, condition, severity):
        """Generate recommendations based on condition and severity"""
        # Basic recommendations based on severity
        if severity >= 4:
            return [
                "Seek emergency medical attention immediately",
                "Call emergency services or go to the nearest emergency room"
            ]
        elif severity == 3:
            return [
                "Consult a healthcare provider within 24 hours",
                "Monitor symptoms closely for any changes",
                "Rest and stay hydrated"
            ]
        elif severity == 2:
            return [
                "Consider scheduling an appointment with your doctor",
                "Rest and monitor your symptoms",
                "Over-the-counter medications may help with symptoms (consult with pharmacist)"
            ]
        else:
            return [
                "Rest and monitor your symptoms",
                "Stay hydrated",
                "Over-the-counter remedies may help with symptoms",
                "If symptoms persist for more than a week, consult a healthcare provider"
            ]
    
    def store_feedback(self, diagnosis_id, accurate, actual_condition=''):
        """Store user feedback"""
        feedback = {
            'diagnosis_id': diagnosis_id,
            'accurate': accurate,
            'actual_condition': actual_condition,
            'timestamp': str(uuid.uuid4())  # Using uuid as timestamp for simplicity
        }
        
        # Ensure data directory exists
        if not os.path.exists('data'):
            os.makedirs('data')
        
        # In a real application, you would store this in a database
        # For this example, we'll append to a JSON file
        try:
            with open('data/feedback.json', 'r') as f:
                feedbacks = json.load(f)
        except:
            feedbacks = []
        
        feedbacks.append(feedback)
            
        with open('data/feedback.json', 'w') as f:
            json.dump(feedbacks, f) 