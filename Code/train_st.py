from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# Initialize the model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Read the text corpus for training
with open("Data/Full_Extraction/combined_text_corpus.txt", "r") as f:
    lines = f.readlines()

# Strip newline characters and other possible leading/trailing whitespaces
lines = [line.strip() for line in lines]

# Create InputExamples
examples = []
for i in range(len(lines) - 1):  # we skip the last line to avoid an IndexError
    examples.append(InputExample(texts=[str(lines[i]), str(lines[i+1])], label=1.0))  # using consecutive lines as pairs

# Create a DataLoader for the training examples
train_dataloader = DataLoader(examples, shuffle=True, batch_size=32)

# Define the training loss
train_loss = losses.CosineSimilarityLoss(model)

# Fine-tuning the model
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=3, warmup_steps=100, output_path="Data/Full_Extraction/")

print("Model fine-tuned and saved to 'Data/Full_Extraction/'")
