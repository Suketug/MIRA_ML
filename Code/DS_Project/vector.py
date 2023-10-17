from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np

# Load the CSV data into a Pandas DataFrame
df = pd.read_csv("Data/Extracted_Data/Formatted_Extracted_Fannie_Mae_Glossary.csv")

# Initialize the Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Vectorize the text columns
vectorized_category = model.encode(df['Category'].astype(str).tolist())
vectorized_sub_category = model.encode(df['Sub_Category'].astype(str).tolist())
vectorized_entity = model.encode(df['Entity'].astype(str).tolist())
vectorized_description = model.encode(df['Description'].astype(str).tolist())

# Convert to list of lists for DataFrame compatibility
vectorized_category = vectorized_category.tolist()
vectorized_sub_category = vectorized_sub_category.tolist()
vectorized_entity = vectorized_entity.tolist()
vectorized_description = vectorized_description.tolist()

# Add the vectors to the DataFrame
df['Vector_Category'] = vectorized_category
df['Vector_Sub_Category'] = vectorized_sub_category
df['Vector_Entity'] = vectorized_entity
df['Vector_Description'] = vectorized_description


# Save the updated DataFrame back to a CSV file or upload to Supabase
df.to_csv("Data/Extracted_Data/vectorized2.csv", index=False)

