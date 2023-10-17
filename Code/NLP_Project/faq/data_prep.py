import pandas as pd
from sklearn.model_selection import train_test_split

# Load the data
data_path = "Data/Extracted_Data/NLP_Data/faq/clean_master_with_intent.csv"  # Update this path
df = pd.read_csv(data_path)

# Extract necessary columns for FAQ generation
df_faq = df[['summarized_text', 'FAQ']].dropna()

# Train-Validation-Test Split
train_df, temp_df = train_test_split(df_faq, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Save to disk (Optional)
train_df.to_csv("Data/Extracted_Data/NLP_Data/faq/train_faq.csv", index=False)
val_df.to_csv("Data/Extracted_Data/NLP_Data/faq/val_faq.csv", index=False)
test_df.to_csv("Data/Extracted_Data/NLP_Data/faq/test_faq.csv", index=False)
