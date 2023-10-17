from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Load the data
data_path = "Data/Extracted_Data/NLP_Data/intent/clean_master_with_intent.csv"  # Update this path
df = pd.read_csv(data_path)

# Drop rows with missing values
df.dropna(subset=['FAQ', 'Intent'], inplace=True)

# Duplicate rows for classes with only one instance
class_counts = df['Intent'].value_counts()
single_instance_classes = class_counts[class_counts == 1].index.tolist()
for intent in single_instance_classes:
    single_row = df[df['Intent'] == intent]
    df = pd.concat([df, single_row], ignore_index=True)

# Label Encoding for Intent
labelencoder = LabelEncoder()
df['Intent_Label'] = labelencoder.fit_transform(df['Intent'])

# Train-Validation-Test Split
try:
    train_df, temp_df = train_test_split(df, test_size=0.4, random_state=42)  # Removed stratify parameter
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)  # Removed stratify parameter
    
    # Save to disk (Optional)
    train_df.to_csv("Data/Extracted_Data/NLP_Data/intent/train_intent.csv", index=False)
    val_df.to_csv("Data/Extracted_Data/NLP_Data/intent/val_intent.csv", index=False)
    test_df.to_csv("Data/Extracted_Data/NLP_Data/intent/test_intent.csv", index=False)
except ValueError as e:
    print(f"An error occurred during data splitting: {e}")
