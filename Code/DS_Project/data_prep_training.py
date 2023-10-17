import pandas as pd
import nltk
from nltk.tokenize import word_tokenize

# Download nltk tokenizer resources
nltk.download('punkt')

# Function to tokenize text
def tokenize_text(text):
    if pd.isna(text):
        return []
    return word_tokenize(str(text))

# Read the DataFrame from the Excel file
df = pd.read_excel("Data/Extracted_Data/master.xlsx")

# Columns to tokenize
columns_to_tokenize = ['Category', 'Sub_Category', 'Entity', 'Cleaned_Description', 'summarized_text', 'entities', 'FAQ', 'FAQ Answers', 'Intent']

# Tokenizing specified columns and saving as new columns
for col in columns_to_tokenize:
    df[col + '_tokenized'] = df[col].apply(tokenize_text)

# Save the DataFrame with tokenized columns back to an Excel file
df.to_excel("Data/Extracted_Data/master_tokenized.xlsx", index=False)
