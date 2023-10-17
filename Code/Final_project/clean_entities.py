import pandas as pd
import ast

def preprocess_entities_column(entity_str):
    # Remove the prefix "Entities:" or "ENTITIES:"
    entity_str = entity_str.replace("Entities:", "").replace("ENTITIES:", "").strip()
    
    # Check if the string is in list format, like "[...]"
    try:
        entity_list = ast.literal_eval(entity_str)
        if isinstance(entity_list, list):
            # Convert list back to comma-separated string
            return ", ".join(map(str, entity_list))
    except (ValueError, SyntaxError):
        pass
    
    return entity_str


# Read the Excel file into a DataFrame
file_path = 'Data/Full_Extraction/master_table_data.xlsx'  # Replace with the actual path to your file
df = pd.read_excel(file_path)

# Apply the preprocessing function to the "entities" column
df['entities'] = df['entities'].astype(str).apply(preprocess_entities_column)

# Save the DataFrame back to an Excel file after preprocessing
output_file_path = 'Data/Full_Extraction/master_table_clean.xlsx'  # Replace with the desired output file path
df.to_excel(output_file_path, index=False)
