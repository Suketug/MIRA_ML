from tqdm import tqdm  # Importing the tqdm library for the progress bar
import spacy
import pandas as pd

# Load your trained spaCy model
model_directory = 'Data/Trained_Models/Entity/OLD/model-best'  # Replace with the path to your trained model
nlp = spacy.load(model_directory)

# Load your test CSV (replace with the actual path)
test_csv_path = 'Data/Full_Extraction/final_missing.csv'  # Replace with the path to your test CSV
test_df = pd.read_csv(test_csv_path)

# Initialize an empty list to store the predicted text entities for each row
predicted_text_entities_list = []

# Initialize the progress bar
pbar = tqdm(total=len(test_df))

# Iterate through the DataFrame and populate the predicted text entities list
for index, row in test_df.iterrows():
    try:
        doc = nlp(row['summarized_table_text'])
        predicted_text_entities = [doc.text[ent.start_char:ent.end_char] for ent in doc.ents]
    except Exception as e:
        print(f"An exception occurred at index {index}: {e}")
        predicted_text_entities = ["Error"]

    predicted_text_entities_list.append(predicted_text_entities)
    
    # Update the progress bar
    pbar.update(1)

# Close the progress bar
pbar.close()

# Add the 'predicted_text_entities' column to the DataFrame
test_df['predicted_text_entities'] = predicted_text_entities_list

# Save the DataFrame with predicted text entities (optional)
output_csv_path = 'Data/Full_Extraction/2_table_entities.csv'  # Replace with the path to your output CSV
test_df.to_csv(output_csv_path, index=False)
