
# Importing the required libraries
from transformers import T5ForConditionalGeneration, T5Tokenizer
import pandas as pd
import torch
from tqdm import tqdm

# Function to run inference
def generate_summaries(model_path, inference_data_path, text_column, output_file):
    # Load the saved model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = T5Tokenizer.from_pretrained("t5-base")

    # Load the inference dataset
    inference_df = pd.read_csv(inference_data_path)
    generated_summaries = []

    # Generate summary for each description
    for index, row in tqdm(inference_df.iterrows(), total=inference_df.shape[0]):
        input_text = row[text_column]

        # Tokenize the input text and generate a summary ID
        inputs = tokenizer("summarize: " + input_text, return_tensors="pt", max_length=512, truncation=True)
        with torch.no_grad():  # disable gradient calculation to save memory
            summary_ids = model.generate(inputs.input_ids, num_beams=4, min_length=50, max_length=200, early_stopping=True)

        # Decode the summary ID and append to the list of generated summaries
        generated_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        generated_summaries.append(generated_summary)

    # Save the generated summaries to a CSV file
    output_df = pd.DataFrame({
        text_column: inference_df[text_column],
        "Generated_Summary": generated_summaries
    })
    output_df.to_csv(output_file, index=False)

# Define the paths and column names
model_path = "Data/Extracted_Data/NLP_Data/summarization/New/summarization_model"  # Replace this with your model path
inference_data_path = "Data/Full_Extraction/test.csv"  # Replace this with your inference data path
text_column = "Cleaned_Description"
output_file = "Data/Full_Extraction/Data_extract_Files/generated_summaries.csv"  # Replace this with your desired output file name

# Run the inference
generate_summaries(model_path, inference_data_path, text_column, output_file)