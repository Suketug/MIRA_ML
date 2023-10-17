# Importing the required libraries
import pandas as pd

def merge_csv_files(master_csv_path, missing_entities_csv_path, output_csv_path, key_column='text_chunk_id'):
    """
    Merges two CSV files based on a common key column, updating missing values in the master file
    with corresponding values from the missing_entities file.

    Parameters:
        master_csv_path (str): Path to the master CSV file
        missing_entities_csv_path (str): Path to the CSV file containing missing entities
        output_csv_path (str): Path to save the updated master CSV file
        key_column (str): The column to use as the key for merging (default is 'text_chunk_id')
    
    Returns:
        None: Saves the updated DataFrame as a new CSV file
    """
    
    try:
        # Read the master CSV file, trying with default UTF-8 encoding first
        master_df = pd.read_csv(master_csv_path)
    except UnicodeDecodeError:
        # If reading with UTF-8 encoding fails, try with ISO-8859-1 encoding
        master_df = pd.read_csv(master_csv_path, encoding='ISO-8859-1')
    
    try:
        # Read the missing entities CSV file, trying with default UTF-8 encoding first
        missing_entities_df = pd.read_csv(missing_entities_csv_path)
    except UnicodeDecodeError:
        # If reading with UTF-8 encoding fails, try with ISO-8859-1 encoding
        missing_entities_df = pd.read_csv(missing_entities_csv_path, encoding='ISO-8859-1')
    
    # Merge the DataFrames based on the key column
    # This will update the master_df in place, filling missing values with corresponding values from missing_entities_df
    merged_df = pd.merge(master_df, missing_entities_df, on=key_column, how='left', suffixes=('', '_missing'))
    
    # Fill the missing values in the master DataFrame with values from the missing entities DataFrame
    for column in master_df.columns:
        if column != key_column:
            merged_df[column].fillna(merged_df[f"{column}_missing"], inplace=True)
            # Drop the temporary column from the missing entities DataFrame
            merged_df.drop(f"{column}_missing", axis=1, inplace=True)

    # Save the updated master DataFrame back to a new CSV file
    merged_df.to_csv(output_csv_path, index=False)

# Test the function (Note: Replace these paths with your actual file paths)
merge_csv_files('Data/Full_Extraction/merged_output_1.csv', 'Data/Full_Extraction/2_table_entities.csv', 'Data/Full_Extraction/merged_output_2.csv')
