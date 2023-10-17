from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Load the DataFrame
df = pd.read_excel("Data/Extracted_Data/master_tokenized.xlsx")

# Initialize Sentence Transformer Model with a reduced dimension
model = SentenceTransformer('all-MiniLM-L6-v2')  # You may change the model as needed

# Columns to vectorize
columns_to_vectorize = ['Category', 'Sub_Category', 'Entity', 'Cleaned_Description', 
                        'summarized_text', 'entities', 'FAQ', 'FAQ Answers']

# Vectorize the specified columns and save as new columns
for col in columns_to_vectorize:
    sentence_embeddings = model.encode(df[col].astype(str).values)
    
    # Reduce the dimension to 128 (or any other number you find appropriate)
    reduced_embeddings = sentence_embeddings[:, :128]  
    
    embeddings_df = pd.DataFrame(reduced_embeddings)
    embeddings_df.columns = [col + '_embedding_' + str(i) for i in embeddings_df.columns]
    
    df = pd.concat([df, embeddings_df], axis=1)

# ... (previous code remains the same)

# Uncasing the 'Intent' column to ensure uniformity
df['Intent'] = df['Intent'].str.lower()

# Count the occurrences of each intent
intent_counts = df['Intent'].value_counts()

# Find intents that occur only once
single_occurrence_intents = intent_counts[intent_counts == 1].index.tolist()

# Duplicate rows where the intent occurs only once
duplicated_rows = df[df['Intent'].isin(single_occurrence_intents)]
df = pd.concat([df, duplicated_rows], axis=0).reset_index(drop=True)

# Label encode the Intent column
le = LabelEncoder()
df['Intent_encoded'] = le.fit_transform(df['Intent'])

# Save the DataFrame with vectorized columns back to an Excel file
df.to_excel("Data/Extracted_Data/master_vectorized_reduced.xlsx", index=False)

# Split the data into training, validation, and test sets
# New line without stratify
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Define the feature columns (assuming they all start with 'embedding')
feature_columns = [col for col in train_df.columns if 'embedding' in col]

# Save the datasets to Excel files
train_df.to_excel("Data/Extracted_Data/train_dataset_reduced.xlsx", index=False)
val_df.to_excel("Data/Extracted_Data/val_dataset_reduced.xlsx", index=False)
test_df.to_excel("Data/Extracted_Data/test_dataset_reduced.xlsx", index=False)


# Separate features and labels
X_train = train_df[feature_columns]
Y_train = train_df['Intent_encoded']
X_val = val_df[feature_columns]
Y_val = val_df['Intent_encoded']
X_test = test_df[feature_columns]
Y_test = test_df['Intent_encoded']

# Save the separated features and labels to Excel files
X_train.to_excel("Data/Extracted_Data/Training_Data/X_train_dataset_reduced.xlsx", index=False)
Y_train.to_excel("Data/Extracted_Data/Training_Data/Y_train_dataset_reduced.xlsx", index=False)
X_val.to_excel("Data/Extracted_Data/Training_Data/X_val_dataset_reduced.xlsx", index=False)
Y_val.to_excel("Data/Extracted_Data/Training_Data/Y_val_dataset_reduced.xlsx", index=False)
X_test.to_excel("Data/Extracted_Data/Training_Data/X_test_dataset_reduced.xlsx", index=False)
Y_test.to_excel("Data/Extracted_Data/Training_Data/Y_test_dataset_reduced.xlsx", index=False)
