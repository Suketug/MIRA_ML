import pandas as pd
import json

# Load the CSV file
data_path = "Data/Extracted_Data/NLP_Data/intent/clean_master_with_intent.csv"
df = pd.read_csv(data_path)

# Prepare the data in Rasa NLU format
rasa_data = {"rasa_nlu_data": {"common_examples": []}}

for index, row in df.iterrows():
    example = {
        "text": row['FAQ'],
        "intent": row['Intent'],
        "entities": []
    }
    rasa_data["rasa_nlu_data"]["common_examples"].append(example)

# Save to JSON file
with open("Data/Extracted_Data/NLP_Data/intent/rasa_nlu_data.json", "w", encoding='utf-8') as f:
    json.dump(rasa_data, f, ensure_ascii=False, indent=4)
