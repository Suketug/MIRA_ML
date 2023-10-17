import pandas as pd
from gensim.summarization.summarizer import summarize

# Read your data into a DataFrame named df
df = pd.read_csv('Data/Extracted_Data/sample_test.csv')

def extractive_summarization(text, ratio=0.3):
    try:
        # Check if text is a string and has more than one sentence
        if isinstance(text, str) and text.count('. ') > 1:  
            return summarize(text, ratio=ratio)
        else:
            return text  # Return original text if it's too short to summarize or not a string
    except Exception as e:
        print(f"An exception occurred: {str(e)}")
        return text  # Handle any other exception and return original text

# Applying the function to the 'Description' column in your DataFrame
df['Summarized_Description'] = df['Cleaned_Description'].apply(extractive_summarization)


# Save the DataFrame back to a CSV
df.to_csv('Data/Extracted_Data/Summarized_Clean_Master_data1.csv', index=False)
