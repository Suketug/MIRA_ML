import pandas as pd
from sklearn.model_selection import train_test_split

# Load the data
data_path = "Data/Extracted_Data/NLP_Data/summarization/Old/clean_master_with_intent.csv"  # Update this path
df = pd.read_csv(data_path)

# Extract necessary columns
df_summarization = df[['Cleaned_Description', 'summarized_text']].dropna()

# Train-Validation-Test Split
train_df, temp_df = train_test_split(df_summarization, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Save to disk (Optional)
train_df.to_csv("Data/Extracted_Data/NLP_Data/summarization/Old/train_summarization.csv", index=False)
val_df.to_csv("Data/Extracted_Data/NLP_Data/summarization/Old/val_summarization.csv", index=False)
test_df.to_csv("Data/Extracted_Data/NLP_Data/summarization/Old/test_summarization.csv", index=False)
