import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk import download, pos_tag, word_tokenize
import json
import os
from sentence_transformers import SentenceTransformer
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    download('punkt')
    download('averaged_perceptron_tagger')
    download('wordnet')
except Exception as e:
    logger.warning(f"Error downloading NLTK data: {e}")

class SymptomAnalyzer:
    def __init__(self, data_path):
        """
        Initialize the SymptomAnalyzer with medical knowledge base.
        
        Args:
            data_path (str): Path to the directory containing medical data files
        """
        self.lemmatizer = WordNetLemmatizer()
        self.data_path = data_path
        
        # Load medical data
        self.symptoms, self.conditions, self.mappings = self._load_medical_data()
        
        # Initialize NLP components
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self._fit_vectorizer()
        
        # Initialize sentence transformer for semantic matching
        try:
            self.sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
            self.use_semantic = True
        except Exception as e:
            logger.warning(f"Could not load sentence transformer: {e}")
            self.use_semantic = False
    
    def _load_medical_data(self):
        """Load medical data from JSON files"""
        try:
            with open(os.path.join(self.data_path, 'symptoms.json'), 'r') as f:
                symptoms = json.load(f)
            
            with open(os.path.join(self.data_path, 'conditions.json'), 'r') as f:
                conditions = json.load(f)
            
            with open(os.path.join(self.data_path, 'symptom_condition_mapping.json'), 'r') as f:
                mappings = json.load(f)
            
            return symptoms, conditions, mappings
        except Exception as e:
            logger.error(f"Error loading medical data: {e}")
            # Return default data if files not found
            return self._create_default_data()
    
    def _create_default_data(self):
        """Create default medical data if files not found"""
        symptoms = {
            "headache": {"id": "S1", "name": "Headache", "description": "Pain in the head or upper neck"},
            "fever": {"id": "S2", "name": "Fever", "description": "Elevated body temperature above normal range"},
            "cough": {"id": "S3", "name": "Cough", "description": "Sudden expulsion of air from the lungs"},
            "fatigue": {"id": "S4", "name": "Fatigue", "description": "Extreme tiredness or exhaustion"},
            "nausea": {"id": "S5", "name": "Nausea", "description": "Sensation of unease in the stomach with an urge to vomit"},
            "sore_throat": {"id": "S6", "name": "Sore Throat", "description": "Pain, scratchiness or irritation of the throat"},
            "shortness_of_breath": {"id": "S7", "name": "Shortness of Breath", "description": "Difficulty breathing or dyspnea"},
            "chest_pain": {"id": "S8", "name": "Chest Pain", "description": "Discomfort or pain in the chest area"},
            "abdominal_pain": {"id": "S9", "name": "Abdominal Pain", "description": "Pain felt between the chest and pelvic regions"},
            "muscle_aches": {"id": "S10", "name": "Muscle Aches", "description": "Pain in muscles throughout the body"}
        }
        
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
            "covid19": {
                "id": "C3", 
                "name": "COVID-19",
                "description": "A respiratory disease caused by SARS-CoV-2 virus",
                "severity": 3,
                "triage_level": "Urgent medical attention",
                "medical_specialty": "Infectious Disease"
            },
            "pneumonia": {
                "id": "C4", 
                "name": "Pneumonia",
                "description": "Infection that inflames air sacs in one or both lungs",
                "severity": 3,
                "triage_level": "Urgent medical attention",
                "medical_specialty": "Pulmonology"
            },
            "heart_attack": {
                "id": "C5", 
                "name": "Heart Attack",
                "description": "Blockage of blood flow to the heart muscle",
                "severity": 4,
                "triage_level": "Emergency care",
                "medical_specialty": "Cardiology"
            }
        }
        
        mappings = {
            "common_cold": ["headache", "sore_throat", "cough", "fatigue"],
            "influenza": ["fever", "headache", "muscle_aches", "fatigue", "cough"],
            "covid19": ["fever", "cough", "fatigue", "shortness_of_breath", "muscle_aches"],
            "pneumonia": ["cough", "fever", "shortness_of_breath", "chest_pain", "fatigue"],
            "heart_attack": ["chest_pain", "shortness_of_breath", "nausea", "fatigue"]
        }
        
        return symptoms, conditions, mappings
    
    def _fit_vectorizer(self):
        """Fit the TF-IDF vectorizer on symptom descriptions"""
        symptom_texts = [
            f"{s['name']} {s['description']}" for s in self.symptoms.values()
        ]
        if symptom_texts:
            self.vectorizer.fit(symptom_texts)
    
    def _preprocess_text(self, text):
        """Preprocess text by tokenizing and lemmatizing"""
        tokens = word_tokenize(text.lower())
        tagged = pos_tag(tokens)
        lemmatized = []
        
        for word, tag in tagged:
            if tag.startswith('N'):
                pos = wordnet.NOUN
            elif tag.startswith('V'):
                pos = wordnet.VERB
            elif tag.startswith('R'):
                pos = wordnet.ADV
            elif tag.startswith('J'):
                pos = wordnet.ADJ
            else:
                pos = wordnet.NOUN
                
            lemmatized.append(self.lemmatizer.lemmatize(word, pos))
        
        return ' '.join(lemmatized)
    
    def _match_symptoms(self, user_symptoms):
        """Match user-described symptoms to known medical symptoms"""
        matched_symptoms = []
        unmatched_symptoms = []
        
        for symptom_text in user_symptoms:
            # Preprocess the symptom text
            processed_text = self._preprocess_text(symptom_text)
            
            # Try exact match first
            exact_match = next(
                (s_id for s_id, s in self.symptoms.items() 
                 if s['name'].lower() == processed_text or 
                 processed_text in s['name'].lower()),
                None
            )
            
            if exact_match:
                matched_symptoms.append(exact_match)
                continue
            
            # If no exact match, try semantic similarity
            if self.use_semantic:
                symptom_embeddings = self.sentence_model.encode(
                    [processed_text] + 
                    [f"{s['name']} {s['description']}" for s in self.symptoms.values()]
                )
                
                similarities = cosine_similarity(
                    [symptom_embeddings[0]], 
                    symptom_embeddings[1:]
                )[0]
                
                best_match_idx = np.argmax(similarities)
                best_similarity = similarities[best_match_idx]
                
                if best_similarity > 0.6:  # Threshold for good match
                    matched_symptom_id = list(self.symptoms.keys())[best_match_idx]
                    matched_symptoms.append(matched_symptom_id)
                    continue
            
            # If semantic matching failed or not available, try TF-IDF vectorization
            symptom_vector = self.vectorizer.transform([processed_text])
            symptom_texts = [
                f"{s['name']} {s['description']}" for s in self.symptoms.values()
            ]
            
            if symptom_texts:
                symptom_matrix = self.vectorizer.transform(symptom_texts)
                similarities = cosine_similarity(symptom_vector, symptom_matrix)[0]
                
                best_match_idx = np.argmax(similarities)
                best_similarity = similarities[best_match_idx]
                
                if best_similarity > 0.3:  # Lower threshold for TF-IDF
                    matched_symptom_id = list(self.symptoms.keys())[best_match_idx]
                    matched_symptoms.append(matched_symptom_id)
                    continue
            
            # If still no match, mark as unmatched
            unmatched_symptoms.append(symptom_text)
        
        return matched_symptoms, unmatched_symptoms
    
    def _calculate_condition_scores(self, matched_symptoms):
        """Calculate condition scores based on matched symptoms"""
        condition_scores = {}
        
        for condition_id, symptoms in self.mappings.items():
            # Count matching symptoms
            matching_count = sum(1 for s in matched_symptoms if s in symptoms)
            
            if matching_count > 0:
                # Calculate score based on match ratio and total symptoms
                total_symptoms = len(symptoms)
                match_ratio = matching_count / total_symptoms
                condition_scores[condition_id] = {
                    'score': match_ratio,
                    'matching_symptoms': matching_count,
                    'total_symptoms': total_symptoms
                }
        
        return condition_scores
    
    def _generate_recommendations(self, condition_id):
        """Generate recommendations based on condition"""
        condition = self.conditions.get(condition_id)
        
        if not condition:
            return ["Please consult a healthcare professional for proper diagnosis."]
        
        severity = condition.get('severity', 1)
        
        if severity >= 4:  # Emergency
            return [
                "Seek emergency medical attention immediately",
                "Call emergency services or go to the nearest emergency room"
            ]
        elif severity == 3:  # Urgent
            return [
                "Seek urgent medical attention within 24 hours",
                "Contact your healthcare provider immediately",
                "Monitor symptoms closely for any changes",
                "Rest and stay hydrated"
            ]
        elif severity == 2:  # Moderate
            return [
                "Schedule an appointment with your healthcare provider soon",
                "Rest and monitor your symptoms",
                "Stay hydrated and get adequate rest",
                "Over-the-counter medications may help with symptoms (consult with pharmacist)"
            ]
        else:  # Mild
            return [
                "Self-care is appropriate for these symptoms",
                "Rest and monitor your symptoms",
                "Stay hydrated and get adequate rest",
                "Over-the-counter remedies may help with symptoms",
                "If symptoms persist for more than a week, consult a healthcare provider"
            ]
    
    def analyze_symptoms(self, user_symptoms, user_info=None):
        """
        Analyze user-provided symptoms and return possible conditions
        
        Args:
            user_symptoms (list): List of symptom strings provided by user
            user_info (dict): Optional user information (age, gender, medical history)
            
        Returns:
            dict: Analysis results including possible conditions and recommendations
        """
        # Match symptoms to medical terminology
        matched_symptoms, unmatched_symptoms = self._match_symptoms(user_symptoms)
        
        # Calculate scores for each condition
        condition_scores = self._calculate_condition_scores(matched_symptoms)
        
        # Sort conditions by score
        sorted_conditions = sorted(
            condition_scores.items(), 
            key=lambda x: x[1]['score'], 
            reverse=True
        )
        
        # Generate results
        results = []
        for condition_id, score_data in sorted_conditions[:3]:  # Top 3 conditions
            condition = self.conditions.get(condition_id)
            if condition:
                results.append({
                    'condition_id': condition_id,
                    'name': condition['name'],
                    'description': condition['description'],
                    'probability': round(score_data['score'] * 100),
                    'matching_symptoms': score_data['matching_symptoms'],
                    'total_symptoms': score_data['total_symptoms'],
                    'severity': condition.get('severity', 1),
                    'triage_level': condition.get('triage_level', 'Self-care'),
                    'medical_specialty': condition.get('medical_specialty', 'General Practice')
                })
        
        # If no conditions found or probability too low
        if not results or results[0]['probability'] < 30:
            results.insert(0, {
                'condition_id': 'unknown',
                'name': 'Unspecified Condition',
                'description': 'Your symptoms do not clearly match any specific condition in our database.',
                'probability': 0,
                'matching_symptoms': 0,
                'total_symptoms': 0,
                'severity': 1,
                'triage_level': 'Medical consultation',
                'medical_specialty': 'General Practice'
            })
        
        # Generate recommendations based on top condition
        recommendations = self._generate_recommendations(
            results[0]['condition_id'] if results else 'unknown'
        )
        
        # Identified symptoms in medical terminology
        identified_symptoms = [
            {
                'id': symptom_id,
                'name': self.symptoms[symptom_id]['name'],
                'description': self.symptoms[symptom_id]['description']
            }
            for symptom_id in matched_symptoms
        ]
        
        return {
            'possible_conditions': results,
            'recommendations': recommendations,
            'identified_symptoms': identified_symptoms,
            'unidentified_symptoms': unmatched_symptoms,
            'disclaimer': 'This is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.'
        } 