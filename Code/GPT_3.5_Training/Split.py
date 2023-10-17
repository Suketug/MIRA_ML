import json
import random

# Initialize an empty list to store the structured data
data = []

# Load the structured data from the JSONL file line-by-line
with open("structured_data.jsonl", "r") as f:
    for line in f:
        data.append(json.loads(line))

# Shuffle the data for randomness
random.shuffle(data)

# Calculate the sizes for train, val, and test sets
total_size = len(data)
train_size = int(total_size * 0.7)
val_size = int(total_size * 0.2)
test_size = total_size - train_size - val_size

# Split the data
train_data = data[:train_size]
val_data = data[train_size:train_size + val_size]
test_data = data[train_size + val_size:]

# Save the splits to JSONL files
with open("Data/train_data.jsonl", "w") as f:
    for entry in train_data:
        f.write(json.dumps(entry) + "\n")

with open("Data/val_data.jsonl", "w") as f:
    for entry in val_data:
        f.write(json.dumps(entry) + "\n")

with open("Data/test_data.jsonl", "w") as f:
    for entry in test_data:
        f.write(json.dumps(entry) + "\n")

print("Data successfully split and saved to train_data.jsonl, val_data.jsonl, and test_data.jsonl.")
