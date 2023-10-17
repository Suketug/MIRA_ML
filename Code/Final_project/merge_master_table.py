import pandas as pd

# Load the data
master_data = pd.read_excel('Data/Full_Extraction/Preprocessed_Master_Fannie_Freddie.xlsx')
table_data = pd.read_excel('Data/Full_Extraction/updated_table_data.xlsx')

# Initialize an empty list to hold the rows for the merged data
merged_data_rows = []

# Iterate through each row in the master data
for i, row in master_data.iterrows():
    # Append the row from the master data
    merged_data_rows.append(row)
    
    # Find the corresponding row in the table data
    table_row = table_data[table_data['text_chunk_id'] == row['text_chunk_id']]
    
    if not table_row.empty:
        # Create a new row based on the table data
        new_row = row.copy()
        new_row['Cleaned_Description'] = table_row.iloc[0]['table_text']
        new_row['summarized_text'] = table_row.iloc[0]['summarized_table_text']
        new_row['entities'] = table_row.iloc[0]['table_entities']
        new_row['FAQ'] = table_row.iloc[0]['table_FAQ']
        new_row['Intent'] = table_row.iloc[0]['table_Intent']

        # Append this new row to the list
        merged_data_rows.append(new_row)

# Create a DataFrame from the list of rows
merged_data = pd.DataFrame(merged_data_rows)

# Save the merged data to a new Excel file
merged_data.to_excel('Data/Full_Extraction/Merged_Master_Fannie_Freddie_Final.xlsx', index=False)
