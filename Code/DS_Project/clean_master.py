import pandas as pd
import re

# Load your DataFrame
df = pd.read_csv("Data/Extracted_Data/master_with_intent.csv")

# Removing the prefix "intents:" or "Intents:"
df['Intent'] = df['Intent'].str.replace('intents:', '', case=False).str.replace('Intents:', '', case=False).str.replace('INTENTS:', '', case=False).str.replace('primary_', '', case=False)

# Save the updated DataFrame to a new CSV file
df.to_csv("Data/Extracted_Data/clean_master_with_intent.csv", index=False)
