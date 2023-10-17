from transformers import T5ForConditionalGeneration, T5Tokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from torch.utils.data import Dataset
import pandas as pd

class CustomTextDataset(Dataset):
    def __init__(self, tokenizer, data_file, text_col, target_col, max_length=512):
        self.tokenizer = tokenizer
        self.data = pd.read_csv(data_file)
        self.text_col = text_col
        self.target_col = target_col
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data.loc[index, self.text_col]
        target = self.data.loc[index, self.target_col]

        # Tokenize text for encoder input
        encoder_inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation="only_first",
            padding="max_length",
            return_attention_mask=True,
            return_tensors='pt'
        )

        # Tokenize target for decoder input
        decoder_inputs = self.tokenizer(
            target,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoder_inputs['input_ids'].squeeze(),
            'attention_mask': encoder_inputs['attention_mask'].squeeze(),
            'decoder_input_ids': decoder_inputs['input_ids'].squeeze(),
            'decoder_attention_mask': decoder_inputs['attention_mask'].squeeze(),
            'labels': decoder_inputs['input_ids'].squeeze()
        }



# Initialize the T5 base model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Initialize custom dataset
train_dataset = CustomTextDataset(tokenizer, "Data/Extracted_Data/NLP_Data/train_summarization.csv", "Cleaned_Description", "summarized_text")
val_dataset = CustomTextDataset(tokenizer, "Data/Extracted_Data/NLP_Data/val_summarization.csv", "Cleaned_Description", "summarized_text")
test_dataset = CustomTextDataset(tokenizer, "Data/Extracted_Data/NLP_Data/test_summarization.csv", "Cleaned_Description", "summarized_text")

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model
)

# Initialize Trainer with TrainingArguments and model
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train
trainer.train()

# Save the model
model.save_pretrained("Data/Trained_Models/summarization_model")

# Evaluate on test data
test_results = trainer.evaluate(test_dataset=test_dataset)

print("Test Results:", test_results)
