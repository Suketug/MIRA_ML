!pip install transformers
!pip install pandas
!pip install torch torchvision
!pip install sentencepiece

# Importing the required libraries
from transformers import T5ForConditionalGeneration, T5Tokenizer
import pandas as pd
import torch

# Function to run inference for FAQ generation
def generate_faq(model_path, inference_data_path, text_column, output_file):
    # Load the saved model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = T5Tokenizer.from_pretrained("t5-base")

    # Load the inference dataset
    inference_df = pd.read_csv(inference_data_path)
    generated_faqs = []

    # Generate FAQ for each summarized text
    for index, row in inference_df.iterrows():
        input_text = row[text_column]

        # Tokenize the input text and generate an FAQ ID
        inputs = tokenizer("generate: " + input_text, return_tensors="pt", max_length=512, truncation=True)
        with torch.no_grad():  # disable gradient calculation to save memory
            faq_ids = model.generate(inputs.input_ids, num_beams=4, min_length=30, max_length=100, early_stopping=True)

        # Decode the FAQ ID and append to the list of generated FAQs
        generated_faq = tokenizer.decode(faq_ids[0], skip_special_tokens=True)
        generated_faqs.append(generated_faq)

    # Save the generated FAQs to a CSV file
    output_df = pd.DataFrame({
        text_column: inference_df[text_column],
        "Generated_FAQ": generated_faqs
    })
    output_df.to_csv(output_file, index=False)

# Define the paths and column names
model_path = "path/to/your/faq_model"  # Replace this with your model path
inference_data_path = "path/to/your/test_faq.csv"  # Replace this with your inference data path
text_column = "summarized_text"
output_file = "path/to/save/generated_faqs.csv"  # Replace this with your desired output file name

# Run the inference
generate_faq(model_path, inference_data_path, text_column, output_file)
