import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
import json
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download required NLTK data with error handling
def download_nltk_data():
    """Download required NLTK data with proper error handling"""
    nltk_resources = ['punkt', 'wordnet', 'averaged_perceptron_tagger']
    for resource in nltk_resources:
        try:
            nltk.download(resource, quiet=True)
            logger.info(f"Successfully downloaded NLTK resource: {resource}")
        except Exception as e:
            logger.warning(f"Error downloading NLTK resource '{resource}': {e}")
            logger.warning(f"Symptom analysis may not work optimally without '{resource}'")

# Try to download NLTK data
download_nltk_data()

class SymptomAnalyzer:
    def __init__(self, data_path=None):
        """
        Initialize the SymptomAnalyzer with medical knowledge base.
        
        Args:
            data_path (str, optional): Path to the directory containing medical data files.
                                     If None, uses default data.
        """
        self.data_path = data_path or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'medical')
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_path, exist_ok=True)
        
        # Initialize NLTK components
        try:
            self.lemmatizer = WordNetLemmatizer()
        except Exception as e:
            logger.warning(f"Error initializing WordNetLemmatizer: {e}")
            logger.warning("Falling back to basic string processing")
            self.lemmatizer = None
            
        # Load medical data
        self.symptoms, self.conditions, self.mappings = self._load_data()
        
        # Initialize vectorizers and models
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self._fit_vectorizer()
        
        # Flag for semantic matching
        self.use_semantic = False
        try:
            # Only import if available, but don't require it
            from sentence_transformers import SentenceTransformer
            self.sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
            self.use_semantic = True
            logger.info("Semantic matching enabled using sentence-transformers")
        except Exception as e:
            logger.warning(f"Could not load sentence transformer: {e}")
            logger.warning("Falling back to TF-IDF similarity matching")
    
    def _load_data(self, data_path=None):
        """
        Load medical conditions, symptoms, and their mappings.
        If a data path is specified, attempts to load from there.
        Otherwise creates default data.
        """
        if data_path:
            try:
                logger.info(f"Attempting to load medical data from {data_path}")
                with open(data_path, 'r') as f:
                    data = json.load(f)
                    symptoms = data.get('symptoms', {})
                    conditions = data.get('conditions', {})
                    mappings = data.get('mappings', {})
                    logger.info(f"Loaded data with {len(symptoms)} symptoms and {len(conditions)} conditions")
                    
                    # Add logging to verify loaded data
                    logger.debug(f"Sample symptoms: {list(symptoms.keys())[:3]}")
                    logger.debug(f"Sample conditions: {list(conditions.keys())[:3]}")
                    logger.debug(f"Sample mappings: {list(mappings.keys())[:3]}")
                    
                    return symptoms, conditions, mappings
            except Exception as e:
                logger.error(f"Failed to load medical data from file: {e}")
                logger.info("Falling back to default medical data")
        
        # If no data path provided or loading failed, create default data
        logger.info("Creating default medical data")
        
        # Default symptoms
        symptoms = {
            "fever": {"id": "S1", "name": "Fever", "description": "Elevated body temperature above the normal range"},
            "cough": {"id": "S2", "name": "Cough", "description": "Sudden expulsion of air from the lungs"},
            "fatigue": {"id": "S3", "name": "Fatigue", "description": "Extreme tiredness resulting from mental or physical exertion"},
            "headache": {"id": "S4", "name": "Headache", "description": "Pain in the head or upper neck"},
            "sore_throat": {"id": "S5", "name": "Sore Throat", "description": "Pain or irritation in the throat"},
            "shortness_of_breath": {"id": "S6", "name": "Shortness of Breath", "description": "Difficulty breathing or feeling breathless"},
            "nausea": {"id": "S7", "name": "Nausea", "description": "Sensation of unease and discomfort in the stomach with urge to vomit"},
            "chest_pain": {"id": "S8", "name": "Chest Pain", "description": "Discomfort or pain in the chest area"},
            "abdominal_pain": {"id": "S9", "name": "Abdominal Pain", "description": "Pain felt between the chest and pelvic regions"},
            "muscle_aches": {"id": "S10", "name": "Muscle Aches", "description": "Pain in muscles throughout the body"},
            # Additional symptoms for better matching
            "runny_nose": {"id": "S11", "name": "Runny Nose", "description": "Excess discharge of fluid from the nose"},
            "dizziness": {"id": "S12", "name": "Dizziness", "description": "Feeling lightheaded or unsteady"},
            "vomiting": {"id": "S13", "name": "Vomiting", "description": "Forceful expulsion of stomach contents through the mouth"},
            "diarrhea": {"id": "S14", "name": "Diarrhea", "description": "Loose, watery, and possibly more frequent bowel movements"},
            "rash": {"id": "S15", "name": "Rash", "description": "Area of irritated or swollen skin that can be itchy, red, and painful"},
            # Add more symptoms with common aliases
            "cold": {"id": "S16", "name": "Common Cold Symptoms", "description": "Symptoms associated with a cold like runny nose and congestion"},
            "chills": {"id": "S17", "name": "Chills", "description": "Feeling of coldness with shivering"},
            "sneezing": {"id": "S18", "name": "Sneezing", "description": "Involuntary expulsion of air from the nose"},
            "congestion": {"id": "S19", "name": "Nasal Congestion", "description": "Stuffy or blocked nose"},
            "joint_pain": {"id": "S20", "name": "Joint Pain", "description": "Pain in one or more joints"},
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
                "description": "Blockage of blood flow to the heart",
                "severity": 4,
                "triage_level": "Emergency",
                "medical_specialty": "Cardiology"
            },
            "migraine": {
                "id": "C6", 
                "name": "Migraine",
                "description": "Recurring type of headache that causes moderate to severe pain",
                "severity": 2,
                "triage_level": "Non-urgent medical care",
                "medical_specialty": "Neurology"
            },
            "allergies": {
                "id": "C7", 
                "name": "Seasonal Allergies",
                "description": "Immune system reaction to environmental triggers",
                "severity": 1,
                "triage_level": "Self-care",
                "medical_specialty": "Allergy and Immunology"
            },
            "gastroenteritis": {
                "id": "C8", 
                "name": "Gastroenteritis",
                "description": "Inflammation of the stomach and intestines",
                "severity": 2,
                "triage_level": "Non-urgent medical care",
                "medical_specialty": "Gastroenterology"
            },
            "fever_condition": {
                "id": "C9", 
                "name": "Fever of Unknown Origin",
                "description": "Elevated body temperature without an immediately obvious cause",
                "severity": 2,
                "triage_level": "Non-urgent medical care",
                "medical_specialty": "General Practice"
            }
        }
        
        # Symptom-to-condition mappings (expanded for better matching)
        mappings = {
            "common_cold": ["cold", "headache", "cough", "sore_throat", "runny_nose", "sneezing", "congestion", "fatigue"],
            "influenza": ["fever", "cough", "fatigue", "muscle_aches", "headache", "chills", "sore_throat"],
            "covid19": ["fever", "cough", "fatigue", "shortness_of_breath", "muscle_aches", "headache", "sore_throat", "diarrhea"],
            "pneumonia": ["cough", "fever", "shortness_of_breath", "chest_pain", "fatigue"],
            "heart_attack": ["chest_pain", "shortness_of_breath", "nausea", "fatigue", "dizziness"],
            "migraine": ["headache", "nausea", "fatigue", "dizziness"],
            "allergies": ["sneezing", "runny_nose", "congestion", "headache", "cough", "fatigue"],
            "gastroenteritis": ["abdominal_pain", "nausea", "vomiting", "diarrhea", "fever"],
            "fever_condition": ["fever", "fatigue", "headache", "chills"]
        }
        
        logger.info("Created default medical data with 20 symptoms and 9 conditions")
        return symptoms, conditions, mappings
    
    def _fit_vectorizer(self):
        """Fit the TF-IDF vectorizer on symptom descriptions"""
        symptom_texts = [
            f"{s['name']} {s['description']}" for s in self.symptoms.values()
        ]
        if symptom_texts:
            self.vectorizer.fit(symptom_texts)
            logger.info("Fitted TF-IDF vectorizer on symptom corpus")
    
    def _get_wordnet_pos(self, tag):
        """Map POS tag to WordNet POS tag for lemmatization"""
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            # Default is noun
            return wordnet.NOUN
    
    def _preprocess_text(self, text):
        """
        Preprocess text using NLTK for tokenization, POS tagging, and lemmatization
        Falls back to simple string processing if NLTK resources are unavailable
        """
        if self.lemmatizer is None:
            # Fallback to basic preprocessing if NLTK is unavailable
            return text.lower().strip()
        
        try:
            # Tokenize text
            tokens = word_tokenize(text.lower())
            
            # POS tagging
            tagged_tokens = pos_tag(tokens)
            
            # Lemmatization with appropriate POS
            lemmatized_tokens = [
                self.lemmatizer.lemmatize(word, self._get_wordnet_pos(tag))
                for word, tag in tagged_tokens
            ]
            
            # Join tokens back into string
            processed_text = ' '.join(lemmatized_tokens)
            return processed_text
        
        except Exception as e:
            logger.warning(f"Error in text preprocessing: {e}")
            logger.warning("Falling back to basic preprocessing")
            # Fallback to basic preprocessing
            return text.lower().strip()
    
    def _match_symptoms(self, user_symptoms):
        """Match user-described symptoms to known medical symptoms"""
        matched_symptoms = []
        unmatched_symptoms = []
        
        # Log the user symptoms for debugging
        logger.info(f"Matching user symptoms: {user_symptoms}")
        
        # Create a lookup for common alternative symptom names
        symptom_aliases = {
            "cold": ["common cold", "runny nose", "congestion", "sneezing"],
            "fever": ["high temperature", "hot", "burning up", "high fever", "temperature"],
            "headache": ["head pain", "migraine", "head hurts", "head ache"],
            "cough": ["coughing", "dry cough", "wet cough", "hack"],
            "fatigue": ["tired", "exhausted", "no energy", "weakness", "low energy"],
            "sore throat": ["throat pain", "throat ache", "painful throat", "throat irritation"],
            "nausea": ["sick to stomach", "queasy", "want to vomit", "stomach upset"],
            "vomiting": ["throwing up", "puking", "emesis", "being sick"],
            "diarrhea": ["loose stool", "watery stool", "runs", "loose bowel"],
            "muscle aches": ["muscle pain", "body aches", "sore muscles", "pain in muscles"],
            "chills": ["shivering", "feeling cold", "goosebumps", "cold sensation"],
            "dizziness": ["lightheaded", "vertigo", "spinning", "unsteady"],
            "chest pain": ["chest discomfort", "chest tightness", "pain in chest"],
            "shortness of breath": ["breathlessness", "can't breathe", "difficulty breathing", "hard to breathe"]
        }
        
        # Add mappings directly from condition names to their associated symptoms
        condition_to_symptoms = {
            "cold": "cold",
            "common cold": "cold"
        }
        
        for symptom_text in user_symptoms:
            # Skip empty symptoms
            if not symptom_text.strip():
                continue
                
            # Process the symptom text
            processed_text = self._preprocess_text(symptom_text)
            logger.debug(f"Processed symptom text: '{processed_text}'")
            
            # Special case for "cold" which is both a condition and a symptom
            if processed_text.lower() == "cold" or "cold" in processed_text.lower().split():
                logger.debug(f"Found special case match for 'cold'")
                matched_symptoms.append("cold")
                continue
                
            # Try exact match first (case insensitive)
            exact_match = None
            for s_id, s in self.symptoms.items():
                if (s['name'].lower() == processed_text or 
                    processed_text in s['name'].lower() or
                    s['id'].lower() == processed_text):
                    exact_match = s_id
                    break
                    
            if exact_match:
                logger.debug(f"Found exact match for '{processed_text}': {exact_match}")
                matched_symptoms.append(exact_match)
                continue
                
            # Check if the symptom corresponds to a condition name
            condition_match = None
            for cond_name, symptom_id in condition_to_symptoms.items():
                if processed_text.lower() == cond_name.lower() or cond_name.lower() in processed_text.lower():
                    condition_match = symptom_id
                    break
                    
            if condition_match:
                logger.debug(f"Found condition-to-symptom match for '{processed_text}': {condition_match}")
                matched_symptoms.append(condition_match)
                continue
                
            # Try matching with common aliases
            alias_match = None
            for key, aliases in symptom_aliases.items():
                if processed_text in aliases or any(alias.lower() in processed_text for alias in aliases):
                    # Find the symptom that best matches this alias
                    for s_id, s in self.symptoms.items():
                        if key.lower() in s['name'].lower():
                            alias_match = s_id
                            break
                    if alias_match:
                        break
                        
            if alias_match:
                logger.debug(f"Found alias match for '{processed_text}': {alias_match}")
                matched_symptoms.append(alias_match)
                continue
                
            # Try partial keyword matching (more lenient)
            keyword_match = None
            for s_id, s in self.symptoms.items():
                symptom_name = s['name'].lower()
                symptom_desc = s['description'].lower()
                
                # Check if the processed text is part of the symptom name or description
                if (processed_text in symptom_name or 
                    any(word in symptom_name for word in processed_text.split()) or
                    processed_text in symptom_desc):
                    keyword_match = s_id
                    break
                    
            if keyword_match:
                logger.debug(f"Found keyword match for '{processed_text}': {keyword_match}")
                matched_symptoms.append(keyword_match)
                continue
                
            # Try semantic matching with sentence transformers if available (more lenient threshold)
            if self.use_semantic:
                try:
                    symptom_embeddings = self.sentence_model.encode(
                        [processed_text] + 
                        [f"{s['name']} {s['description']}" for s in self.symptoms.values()]
                    )
                    
                    similarities = cosine_similarity(
                        [symptom_embeddings[0]], 
                        symptom_embeddings[1:]
                    )[0]
                    
                    best_match_idx = np.argmax(similarities)
                    # Lower threshold for better match rate
                    if similarities[best_match_idx] > 0.35:
                        semantic_match = list(self.symptoms.keys())[best_match_idx]
                        logger.debug(f"Found semantic match for '{processed_text}': {semantic_match}")
                        matched_symptoms.append(semantic_match)
                        continue
                except Exception as e:
                    logger.warning(f"Error in semantic matching: {e}")
            
            # Try TF-IDF similarity as a fallback
            try:
                # Convert symptom to vector
                symptom_vector = self.vectorizer.transform([processed_text])
                
                # Get vectors for all known symptoms
                symptom_texts = [
                    f"{s['name']} {s['description']}" for s in self.symptoms.values()
                ]
                all_symptom_vectors = self.vectorizer.transform(symptom_texts)
                
                # Calculate similarity
                similarities = cosine_similarity(symptom_vector, all_symptom_vectors)[0]
                
                # Get best match if similarity is above threshold
                best_match_idx = np.argmax(similarities)
                # Lower threshold for better match rate
                if similarities[best_match_idx] > 0.2:
                    tfidf_match = list(self.symptoms.keys())[best_match_idx]
                    logger.debug(f"Found TF-IDF match for '{processed_text}': {tfidf_match}")
                    matched_symptoms.append(tfidf_match)
                else:
                    unmatched_symptoms.append(symptom_text)
            except Exception as e:
                logger.warning(f"Error in TF-IDF matching: {e}")
                unmatched_symptoms.append(symptom_text)
        
        # Log the matched and unmatched symptoms
        logger.info(f"Matched symptoms: {matched_symptoms}")
        logger.info(f"Unmatched symptoms: {unmatched_symptoms}")
        
        return matched_symptoms, unmatched_symptoms

    def analyze(self, symptoms, user_info=None):
        """
        Analyze a list of symptoms and return possible conditions.
        
        Args:
            symptoms (list): List of symptom descriptions
            user_info (dict, optional): User information like age, gender, etc.
            
        Returns:
            dict: Analysis results with possible conditions and recommendations
        """
        logger.info(f"Analyzing symptoms: {symptoms}")
        
        if not symptoms:
            logger.warning("Empty symptoms list provided")
            return {
                "possible_conditions": [],
                "recommendations": ["Please provide symptoms for analysis"],
                "identified_symptoms": [],
                "unidentified_symptoms": []
            }
        
        # Match symptoms to known medical terms
        matched_symptoms, unmatched = self._match_symptoms(symptoms)
        
        if not matched_symptoms:
            logger.warning(f"No symptoms matched from input: {symptoms}")
            return {
                "possible_conditions": [],
                "recommendations": [
                    "Could not match your symptoms to known conditions",
                    "Please consult a healthcare provider for proper diagnosis"
                ],
                "identified_symptoms": [],
                "unidentified_symptoms": unmatched
            }
        
        # Find conditions that match the symptoms
        condition_matches = {}
        
        logger.info(f"Looking for conditions matching symptoms: {matched_symptoms}")
        for condition_id, symptom_list in self.mappings.items():
            # Count how many symptoms match
            matches = set(matched_symptoms).intersection(set(symptom_list))
            match_count = len(matches)
            
            # Log all potential matches for debugging
            if match_count > 0:
                logger.debug(f"Condition {condition_id} matches {match_count} symptoms: {matches}")
            
            # Record matches if there are any
            if match_count > 0:
                # Calculate match percentage
                match_percentage = match_count / len(symptom_list)
                # Boost the match percentage if we have a high proportion of the condition's symptoms
                if match_count / len(symptom_list) > 0.5:
                    match_percentage += 0.1
                    
                # Ensure condition exists in our database
                if condition_id not in self.conditions:
                    logger.warning(f"Condition {condition_id} referenced in mappings but not found in conditions database")
                    continue
                    
                condition_matches[condition_id] = {
                    "match_count": match_count,
                    "match_percentage": match_percentage,
                    "condition": self.conditions[condition_id],
                    "matching_symptoms": list(matches)
                }
                logger.debug(f"Condition match: {condition_id}, count: {match_count}, percentage: {match_percentage}")
        
        # If no conditions were matched but we have symptoms, try a more lenient matching approach
        if not condition_matches and matched_symptoms:
            logger.info("No direct condition matches found, trying more lenient matching")
            for condition_id, condition in self.conditions.items():
                # Check if any words in the condition name match our symptoms
                condition_name_lower = condition['name'].lower()
                for symptom_id in matched_symptoms:
                    # Get the actual symptom name (not ID)
                    symptom_name = self.symptoms[symptom_id]['name'].lower()
                    
                    # Check for overlap between symptom name and condition name
                    if (symptom_name in condition_name_lower or 
                        any(word in condition_name_lower for word in symptom_name.split())):
                        
                        logger.debug(f"Found fuzzy match: symptom '{symptom_name}' matches condition '{condition['name']}'")
                        
                        condition_matches[condition_id] = {
                            "match_count": 1,
                            "match_percentage": 0.3,  # Conservative match percentage
                            "condition": condition,
                            "matching_symptoms": [symptom_id]
                        }
        
        # Sort conditions by match percentage
        sorted_conditions = sorted(
            condition_matches.values(),
            key=lambda x: x["match_percentage"],
            reverse=True
        )
        
        # Create response
        possible_conditions = [
            {
                "name": c["condition"]["name"],
                "description": c["condition"]["description"],
                "probability": round(c["match_percentage"], 2),
                "severity": c["condition"].get("severity", 1),
                "matching_symptom_count": c["match_count"],
                "matching_symptoms": [self.symptoms[symptom_id]["name"] for symptom_id in c["matching_symptoms"]]
            }
            for c in sorted_conditions[:3]  # Top 3 matches
        ]
        
        # Generate recommendations based on top condition severity
        recommendations = ["Consult a healthcare provider for proper diagnosis"]
        
        if sorted_conditions:
            top_condition = sorted_conditions[0]["condition"]
            severity = top_condition.get("severity", 1)
            
            if severity == 1:
                recommendations = [
                    "Rest and stay hydrated",
                    "Over-the-counter medications may help with symptoms",
                    "Monitor your symptoms for any worsening"
                ]
            elif severity == 2:
                recommendations = [
                    "Rest and stay hydrated",
                    "Consider scheduling a non-urgent appointment with your doctor",
                    "Monitor your symptoms closely for any changes"
                ]
            elif severity == 3:
                recommendations = [
                    "Seek medical attention soon",
                    "Your symptoms may require professional evaluation",
                    "Stay hydrated and rest until you can see a healthcare provider"
                ]
            else:  # severity >= 4
                recommendations = [
                    "Seek immediate medical attention",
                    "Your symptoms may indicate a serious condition",
                    "Please go to the nearest emergency room or call emergency services"
                ]
        
        # Prepare identified symptoms for response
        identified_symptoms = [
            self.symptoms[symptom_id]["name"]
            for symptom_id in matched_symptoms
        ]
        
        return {
            "possible_conditions": possible_conditions,
            "recommendations": recommendations,
            "identified_symptoms": identified_symptoms,
            "unidentified_symptoms": unmatched,
            "disclaimer": "This analysis is for informational purposes only and does not constitute medical advice. Always consult with a qualified healthcare provider for diagnosis and treatment."
        }