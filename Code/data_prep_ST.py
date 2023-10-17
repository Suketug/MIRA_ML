import pandas as pd

def combine_columns_for_finetuning(file_path, columns_to_combine):
    # Load the Excel file
    df = pd.read_excel(file_path)
    
    # Combine the selected columns
    df['combined_text'] = df[columns_to_combine].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    
    # Save the combined text corpus to a new file
    combined_file_path = "Data/Full_Extraction/combined_text_corpus.txt"
    df['combined_text'].to_csv(combined_file_path, index=False, header=False)
    
    print(f"Combined text corpus saved to: {combined_file_path}")

# Columns you want to combine
columns_to_combine = ['Cleaned_Description', 'summarized_text', 'entities', 'FAQ', 'Intent']

# Path to your Excel file
file_path = "Data/Full_Extraction/Merged_Master_Fannie_Freddie_Final.xlsx"  # Replace with your actual file path

# Execute the function
combine_columns_for_finetuning(file_path, columns_to_combine)
