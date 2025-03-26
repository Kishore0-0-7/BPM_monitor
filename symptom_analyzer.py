import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import uuid
import json
import os
import traceback

class SymptomAnalyzer:
    def __init__(self):
        print("Starting SymptomAnalyzer initialization...")
        try:
            # Download necessary NLTK data
            print("Downloading NLTK data...")
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            
            # Initialize lemmatizer and stopwords
            print("Initializing NLP tools...")
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
            
            # Load the dataset of symptoms and conditions
            print("Loading data...")
            self.load_data()
            
            # Train the model
            print("Training model...")
            self.train_model()
            print("SymptomAnalyzer initialization complete!")
        except Exception as e:
            print(f"Error in SymptomAnalyzer initialization: {str(e)}")
            print(traceback.format_exc())
            raise
    
    def load_data(self):
        """Load symptom-disease dataset"""
        # In a real application, you would have a comprehensive dataset
        # For this example, we'll create a simple dataset
        try:
            self.data = pd.read_csv('data/symptom_disease.csv')
        except:
            # Create sample data if file doesn't exist
            self.create_sample_data()
            self.data = pd.read_csv('data/symptom_disease.csv')
    
    def create_sample_data(self):
        """Create a sample dataset for demonstration"""
        if not os.path.exists('data'):
            os.makedirs('data')
        
        # Sample data mapping symptoms to conditions
        data = {
            'symptoms': [
                'fever headache fatigue',
                'cough fever difficulty breathing',
                'rash itching redness',
                'abdominal pain nausea vomiting',
                'sore throat runny nose cough',
                'chest pain shortness of breath',
                'joint pain swelling stiffness',
                'frequent urination excessive thirst hunger',
                'headache sensitivity to light nausea',
                'fever rash joint pain'
            ],
            'condition': [
                'Common Cold',
                'Pneumonia',
                'Dermatitis',
                'Gastroenteritis',
                'Upper Respiratory Infection',
                'Angina',
                'Arthritis',
                'Diabetes',
                'Migraine',
                'Dengue Fever'
            ],
            'severity': [1, 3, 2, 2, 1, 4, 2, 3, 2, 3]
        }
        
        pd.DataFrame(data).to_csv('data/symptom_disease.csv', index=False)
    
    def preprocess_text(self, text):
        """Preprocess symptom text"""
        # Tokenize, remove stopwords, and lemmatize
        words = text.lower().split()
        words = [self.lemmatizer.lemmatize(w) for w in words if w not in self.stop_words]
        return ' '.join(words)
    
    def train_model(self):
        """Train the ML model on the symptom-disease dataset"""
        # Preprocess symptoms
        self.data['processed_symptoms'] = self.data['symptoms'].apply(self.preprocess_text)
        
        # Create TF-IDF features from symptoms
        self.vectorizer = TfidfVectorizer(max_features=5000)
        X = self.vectorizer.fit_transform(self.data['processed_symptoms'])
        y = self.data['condition']
        
        # Train a Random Forest classifier
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X, y)
    
    def analyze(self, symptoms, user_info=None):
        """
        Analyze the given symptoms and return possible diagnoses
        
        Parameters:
        symptoms (list): List of symptom strings
        user_info (dict): User information (age, gender, medical history)
        
        Returns:
        dict: Analysis results including possible conditions and recommendations
        """
        # Process input symptoms
        symptoms_text = ' '.join(symptoms)
        processed_symptoms = self.preprocess_text(symptoms_text)
        
        # Vectorize the processed symptoms
        symptom_vector = self.vectorizer.transform([processed_symptoms])
        
        # Get predictions and probabilities
        predictions = self.model.predict_proba(symptom_vector)
        
        # Get top 3 possible conditions
        top_indices = np.argsort(predictions[0])[::-1][:3]
        conditions = self.model.classes_[top_indices]
        probabilities = predictions[0][top_indices]
        
        # Create result
        results = []
        for i in range(len(conditions)):
            severity = int(self.data[self.data['condition'] == conditions[i]]['severity'].iloc[0])
            results.append({
                'condition': conditions[i],
                'probability': float(probabilities[i]),
                'severity': severity
            })
        
        # Generate diagnosis ID for feedback
        diagnosis_id = str(uuid.uuid4())
        
        # Generate recommendations based on highest probability condition
        recommendations = self.generate_recommendations(results[0]['condition'], results[0]['severity'])
        
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
        """Store user feedback to improve the model"""
        feedback = {
            'diagnosis_id': diagnosis_id,
            'accurate': accurate,
            'actual_condition': actual_condition,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        # In a real application, you would store this in a database
        # For this example, we'll append to a JSON file
        try:
            with open('data/feedback.json', 'r') as f:
                feedbacks = json.load(f)
        except:
            feedbacks = []
        
        feedbacks.append(feedback)
        
        if not os.path.exists('data'):
            os.makedirs('data')
            
        with open('data/feedback.json', 'w') as f:
            json.dump(feedbacks, f) 