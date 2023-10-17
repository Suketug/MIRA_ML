import pandas as pd
import math

def split_csv_into_smaller_files(input_file_path, output_folder_path, rows_per_file):
    # Read the original CSV
    df = pd.read_csv(input_file_path)

    # Calculate the number of smaller DataFrames needed
    total_rows = df.shape[0]
    num_files = math.ceil(total_rows / rows_per_file)

    # Split the original DataFrame into smaller ones and save each
    for i in range(num_files):
        start_idx = i * rows_per_file
        end_idx = (i + 1) * rows_per_file
        sub_df = df.iloc[start_idx:end_idx]

        # Save the smaller DataFrame to a new CSV file
        sub_df.to_csv(f"{output_folder_path}/missing_entities_part_{i+1}.csv", index=False)

# Usage example
split_csv_into_smaller_files(
    input_file_path="Data/Extracted_Data/master.csv",
    output_folder_path="Data/Extracted_Data/FAQ_CSV's",
    rows_per_file=500
)
