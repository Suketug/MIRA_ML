import pandas as pd
from spacy.lang.en import English
from nltk.corpus import wordnet

# Initialize the spacy tokenizer
nlp = English()

# Load your intent data
input_file_path = 'Data/Extracted_Data/NLP_Data/intent/clean_master_with_intent.csv'  # Replace with your actual file path
df_intent = pd.read_csv(input_file_path)

# Initialize an empty list to store synonyms
synonyms_list = []

# Iterate through the DataFrame and populate the synonyms list
for index, row in df_intent.iterrows():
    intent = row['Intent']
    
    # Skip if the intent is not a string (e.g., NaN)
    if not isinstance(intent, str):
        synonyms_list.append('')
        continue
    
    doc = nlp(intent)
    synonyms = []
    for token in doc:
        for syn in wordnet.synsets(token.text):
            for lemma in syn.lemmas():
                synonyms.append(lemma.name())

    # Remove duplicates by converting list to set and back to list
    synonyms = list(set(synonyms))

    # Add to synonyms list
    synonyms_list.append(', '.join(synonyms))

# Add the 'Synonyms' column to the DataFrame
df_intent['Synonyms'] = synonyms_list

# Save the DataFrame with synonyms (replace with your desired file path)
output_file_path = 'Data/Extracted_Data/NLP_Data/intent/synonymns_intent.csv'  # Replace with your actual file path
df_intent.to_csv(output_file_path, index=False)
