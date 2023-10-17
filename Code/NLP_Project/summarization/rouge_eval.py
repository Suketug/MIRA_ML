import pandas as pd
from rouge import Rouge

# Load the CSV file into a DataFrame
generated_summaries = pd.read_csv("Data/Extracted_Data/NLP_Data/generated_summaries.csv")

# Initialize the Rouge evaluator
rouge = Rouge()

# Use the entire dataset for evaluation
data = generated_summaries

# Calculate Rouge scores
scores = [rouge.get_scores(hyp, ref, avg=True) for hyp, ref in zip(data['Generated_Summary'], data['Cleaned_Description'])]

# Variables to store sum of scores for calculating average
sum_rouge1 = 0
sum_rouge2 = 0
sum_rougeL = 0

# Displaying Rouge scores for the data and summing up the scores
for i, score in enumerate(scores):
    print(f"Sample {i+1}:")
    print(f"Rouge-1: {score['rouge-1']['f']}, Rouge-2: {score['rouge-2']['f']}, Rouge-L: {score['rouge-l']['f']}")
    print("-" * 50)
    
    sum_rouge1 += score['rouge-1']['f']
    sum_rouge2 += score['rouge-2']['f']
    sum_rougeL += score['rouge-l']['f']

# Calculate and display average Rouge scores
total_samples = len(scores)
avg_rouge1 = sum_rouge1 / total_samples
avg_rouge2 = sum_rouge2 / total_samples
avg_rougeL = sum_rougeL / total_samples

print(f"Average Rouge-1: {avg_rouge1}, Average Rouge-2: {avg_rouge2}, Average Rouge-L: {avg_rougeL}")
