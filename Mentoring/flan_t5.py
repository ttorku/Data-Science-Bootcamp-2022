# Filename: flan_t5_fine_tuning.py

import pandas as pd
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from rouge_score import rouge_scorer
import sacrebleu
import torch

# Function to split text into chunks based on a maximum character length
def split_into_chunks(text, max_length):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        word_length = len(word) + 1  # Add 1 for the space
        if current_length + word_length <= max_length:
            current_chunk.append(word)
            current_length += word_length
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = word_length
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

# Prepare data for Flan-T5
def prepare_data_flan_t5(df, max_length=512):
    data = []

    for _, row in df.iterrows():
        input_text = f"Process Name: {row['process_name']}\nProcess Description: {row['process_description']}\nRisk Description: {row['risk_description']}"
        output_text = f"Control Description: {row['control_description']}"
        input_chunks = split_into_chunks(input_text, max_length)
        output_chunks = split_into_chunks(output_text, max_length)
        data.append((input_chunks, output_chunks))

    # Flatten the data to get individual chunks for input and output
    flattened_data = []
    for input_chunks, output_chunks in data:
        for in_chunk in input_chunks:
            for out_chunk in output_chunks:
                flattened_data.append((in_chunk, out_chunk))

    chunked_df = pd.DataFrame(flattened_data, columns=["input_text", "output_text"])
    return chunked_df

# Load and prepare the dataset
df = pd.read_csv("business_processes_dataset.csv")
chunked_df_flan_t5 = prepare_data_flan_t5(df)
chunked_df_flan_t5.to_csv("chunked_data_flan_t5.csv", index=False)

# Load the tokenizer and model
model_name = "google/flan-t5-large"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Prepare the dataset
class FineTuneDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        input_text = self.dataframe.iloc[idx, 0]
        output_text = self.dataframe.iloc[idx, 1]
        inputs = self.tokenizer.encode_plus(input_text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        outputs = self.tokenizer.encode_plus(output_text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        
        input_ids = inputs["input_ids"].squeeze()
        attention_mask = inputs["attention_mask"].squeeze()
        labels = outputs["input_ids"].squeeze()
        
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# Create dataset and data collator
train_dataset = FineTuneDataset(chunked_df_flan_t5, tokenizer, max_length=512)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./flan_t5_fine_tuned",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=500,
    evaluation_strategy="steps",
    save_total_limit=2,
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)

# Fine-tune the model
trainer.train()

# Function to generate control descriptions with chunking
def generate_control_description(input_text):
    input_chunks = split_into_chunks(input_text, 512)
    generated_chunks = []
    
    for chunk in input_chunks:
        inputs = tokenizer(chunk, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
        output_sequences = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_length=512)
        generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
        generated_chunks.append(generated_text)
    
    return ' '.join(generated_chunks)

# Evaluation metrics
def evaluate_model(model, tokenizer, test_df):
    predictions = []
    references = []
    
    for _, row in test_df.iterrows():
        input_text = f"Process Name: {row['process_name']}\nProcess Description: {row['process_description']}\nRisk Description: {row['risk_description']}"
        generated_description = generate_control_description(input_text)
        predictions.append(generated_description)
        references.append(row['control_description'])
    
    # Calculate BLEU score
    bleu_score = sacrebleu.corpus_bleu(predictions, [references])
    
    # Calculate ROUGE scores
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = [scorer.score(ref, pred) for ref, pred in zip(references, predictions)]
    
    # Average ROUGE scores
    avg_rouge1 = sum([score['rouge1'].fmeasure for score in rouge_scores]) / len(rouge_scores)
    avg_rouge2 = sum([score['rouge2'].fmeasure for score in rouge_scores]) / len(rouge_scores)
    avg_rougeL = sum([score['rougeL'].fmeasure for score in rouge_scores]) / len(rouge_scores)
    
    return bleu_score, avg_rouge1, avg_rouge2, avg_rougeL

# Load the test dataset (replace with your test dataset)
test_df = pd.read_csv("chunked_data_flan_t5.csv")

# Evaluate the model
bleu_score, avg_rouge1, avg_rouge2, avg_rougeL = evaluate_model(model, tokenizer, test_df)
print("BLEU Score:", bleu_score.score)
print("Average ROUGE-1 Score:", avg_rouge1)
print("Average ROUGE-2 Score:", avg_rouge2)
print("Average ROUGE-L Score:", avg_rougeL)
