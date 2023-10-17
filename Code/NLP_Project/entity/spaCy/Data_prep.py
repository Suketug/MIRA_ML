import pandas as pd
import re
import spacy
from spacy.tokens import DocBin
from tqdm import tqdm
from spacy.util import filter_spans

# Function to escape special characters in text for regex
def escape_text(text):
    return re.escape(str(text))

# Initialize spaCy
nlp = spacy.blank("en")
doc_bin = DocBin()

# Load the CSV file into a DataFrame
df = pd.read_csv('Data/Extracted_Data/NLP_Data/entity/LSTM/clean_master_with_intent.csv')

# Iterate through each row in the DataFrame
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    text = row['Cleaned_Description']
    entities = str(row['entities']).split(", ")  # Assuming entities are comma-separated
    label = row['Sub_Category']  # Using Sub_Categories as the entity label

    if pd.isna(text) or pd.isna(row['entities']) or pd.isna(label):
        continue

    doc = nlp.make_doc(text)
    ents = []

    # Find the start and end indices of each entity
    for entity in entities:
        escaped_entity = escape_text(entity)
        for match in re.finditer(escaped_entity, str(text)):
            start, end = match.span()
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is None:
                print("Skipping entity")
            else:
                ents.append(span)

    # Filter overlapping entities
    filtered_ents = filter_spans(ents)
    doc.ents = filtered_ents
    doc_bin.add(doc)

# Save to disk
doc_bin.to_disk("Data/spaCy/spacy_train_data.spacy")
