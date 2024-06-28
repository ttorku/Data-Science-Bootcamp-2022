
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

# Load dataset
data = load_dataset('csv', data_files={'train': 'path_to_train_dataset.csv', 'validation': 'path_to_validation_dataset.csv', 'test': 'path_to_test_dataset.csv'})

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenization function
def tokenize_function(example):
    inputs = tokenizer(example['input_sequence'], padding="max_length", truncation=True, max_length=128)
    outputs = tokenizer(example['output_sequence'], padding="max_length", truncation=True, max_length=128)
    inputs['labels'] = outputs['input_ids']
    return inputs

tokenized_datasets = data.map(tokenize_function, batched=True)

# Model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=tokenizer.vocab_size)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Compute metrics
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average='macro')
    acc = accuracy_score(p.label_ids, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Evaluate the model
evaluation_results = trainer.evaluate(tokenized_datasets['test'])
print(evaluation_results)

# Save the model
model.save_pretrained('path_to_save_model')
tokenizer.save_pretrained('path_to_save_tokenizer')

# Deployment example
from transformers import pipeline

# Load the fine-tuned model and tokenizer
model = BertForSequenceClassification.from_pretrained('path_to_save_model')
tokenizer = BertTokenizer.from_pretrained('path_to_save_tokenizer')

# Create a pipeline for inference
nlp = pipeline('text-generation', model=model, tokenizer=tokenizer)

# Generate controls
def generate_control(business_process, risk):
    input_text = business_process + " " + risk
    generated_control = nlp(input_text, max_length=50, num_return_sequences=1)
    return generated_control[0]['generated_text']

# Example usage
business_process = "Example business process"
risk = "Associated risk"
control = generate_control(business_process, risk)
print(f"Generated Control: {control}")
