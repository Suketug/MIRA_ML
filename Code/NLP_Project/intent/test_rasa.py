from rasa.shared.nlu.interpreter import NaturalLanguageInterpreter
from rasa.core.agent import Agent
import pandas as pd

model_path = "Data/Trained_Models/intent/nlu-20230925-120351-bright-library.tar.gz"

# Load the model
interpreter = NaturalLanguageInterpreter.load(model_path)

# Load the CSV file into a DataFrame
df = pd.read_csv("Data/Extracted_Data/NLP_Data/intent/test.csv")

# Create a column to store the predicted intents
df['predicted_intent'] = None

# Iterate through the DataFrame and predict intents
for index, row in df.iterrows():
    message = row['FAQ']  # assuming 'query' is the name of the column containing the text queries
    result = interpreter.parse(message)
    intent = result['intent']['name']
    df.at[index, 'predicted_intent'] = intent

# Save the DataFrame back to CSV
df.to_csv("Data/Extracted_Data/NLP_Data/intent/test_results.csv", index=False)
