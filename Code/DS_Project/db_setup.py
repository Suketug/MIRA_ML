from supabase_py import create_client, Client
import pandas as pd
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get Supabase credentials
supabase_url = os.getenv("SUPABASE_URL")
api_key = os.getenv("SUPABASE_API_KEY")

# Initialize Supabase client
supabase: Client = create_client(supabase_url, api_key)

# Load the vectorized CSV data into a Pandas DataFrame
df = pd.read_csv("Data/Extracted_Data/vectorized2.csv")

# Convert DataFrame to a list of dictionaries (one dictionary per row)
data_to_insert = df.to_dict(orient='records')

# Handle NaN and vector fields
for row in data_to_insert:
    for key, value in row.items():
        if pd.isna(value):
            row[key] = None  # Replace NaN with None

    # Convert vector fields to JSON format (they are already lists)
    for vector_field in ['Vector_Category', 'Vector_Sub_Category', 'Vector_Entity', 'Vector_Description']:
        if vector_field in row and row[vector_field] is not None:
            row[vector_field] = row[vector_field]  # No need to use json.dumps() as they are already lists

# Insert data into Supabase
response = supabase.table('kb_data_1').insert(data_to_insert).execute()

# Check if the insertion was successful
if response.get('error') is None:
    print(f"{len(data_to_insert)} rows inserted successfully.")
else:
    print(f"Failed to insert rows: {response['error']}")
