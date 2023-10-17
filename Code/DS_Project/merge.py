# Import necessary library
import pandas as pd

# Load the data (replace with your actual file path)
df = pd.read_csv("Data/Extracted_Data/master_with_intent.csv")

# Determine the number of unique intents
num_intent_classes = df['Intent'].nunique()

print(f"Number of unique intents: {num_intent_classes}")
