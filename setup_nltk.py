import nltk
import os

# Create data directory if it doesn't exist
if not os.path.exists('data'):
    os.makedirs('data')

# Download necessary NLTK data
print("Downloading NLTK data...")
nltk.download('stopwords')
nltk.download('wordnet')
print("NLTK data download complete!") 