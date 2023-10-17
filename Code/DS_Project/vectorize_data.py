from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np

# Load the DataFrame
df = pd.read_excel("Data/Extracted_Data/master_tokenized.xlsx")

# Initialize Sentence Transformer Model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Columns to vectorize
columns_to_vectorize = ['Category', 'Sub_Category', 'Entity', 'Cleaned_Description', 
                        'summarized_text', 'entities', 'FAQ', 'FAQ Answers', 'Intent']

# Vectorize the specified columns and save as new columns
for col in columns_to_vectorize:
    # Generate sentence embeddings
    sentence_embeddings = model.encode(df[col].astype(str).values)
    
    # Convert embeddings to DataFrame
    embeddings_df = pd.DataFrame(np.array(sentence_embeddings))
    
    # Prefix the embeddings DataFrame columns with original column name
    embeddings_df.columns = [col + '_embedding_' + str(i) for i in embeddings_df.columns]
    
    # Concatenate the original DataFrame with the embeddings DataFrame
    df = pd.concat([df, embeddings_df], axis=1)

# Save the DataFrame with vectorized columns back to an Excel file
df.to_excel("Data/Extracted_Data/master_vectorized.xlsx", index=False)
