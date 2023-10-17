import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from unidecode import unidecode

# Read the dataset
df = pd.read_csv('Data/Extracted_Data/Master_data1.csv')

# Initialize stopwords
stop_words = set(stopwords.words('english'))

# Function to clean text
def clean_text(text):
    if pd.isna(text) or not isinstance(text, str):  # Check if the text is NaN or not a string
        return ''
    text = unidecode(text)  # Handle encoding issues
    text = re.sub(r'[^a-zA-Z0-9.,!?]', ' ', text)  # Keep only essential punctuations
    text = text.lower()  # Convert to lowercase
    text = ' '.join(word for word in text.split() if word not in stop_words)  # Stopword removal
    return text

# Apply the function on the Description column
df['Cleaned_Description'] = df['Description'].apply(clean_text)

# Save the pre-processed data
df.to_csv('Data/Extracted_Data/Clean_Master_data1.csv', index=False)
