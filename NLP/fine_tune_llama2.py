
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset

# Load the model and tokenizer
model_name = "llama2-13b-chat-gptq"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Sample dataset
dataset = [
    {"process": "Order to Cash", "risk": "Credit risk", "control": "Credit limit setting and approval process"},
    {"process": "Procurement", "risk": "Fraud risk", "control": "Vendor verification and approval process"},
    # Add more records...
]

# Convert the dataset into a pandas DataFrame
df = pd.DataFrame(dataset)

# Function to create prompt and target
def create_prompt_target(record):
    prompt = f"Business Process: {record['process']}
Risk: {record['risk']}
Control:"
    target = record['control']
    return {"prompt": prompt, "target": target}

# Create prompts and targets
prompt_target_dataset = df.apply(create_prompt_target, axis=1)

# Tokenization function
def tokenize_function(examples):
    inputs = tokenizer(examples['prompt'], padding="max_length", truncation=True, max_length=256)
    targets = tokenizer(examples['target'], padding="max_length", truncation=True, max_length=256)
    inputs["labels"] = targets["input_ids"]
    return inputs

# Tokenize the dataset
tokenized_dataset = prompt_target_dataset.apply(lambda x: tokenize_function(x), axis=1)

# Convert the tokenized dataset into a format suitable for the Trainer
def format_data(example):
    return {
        'input_ids': torch.tensor(example['input_ids']),
        'attention_mask': torch.tensor(example['attention_mask']),
        'labels': torch.tensor(example['labels'])
    }

# Apply formatting function
formatted_dataset = tokenized_dataset.apply(format_data)

# Convert to Hugging Face Dataset
formatted_dataset = Dataset.from_pandas(pd.DataFrame(list(formatted_dataset)))

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs',
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=formatted_dataset,
    tokenizer=tokenizer
)

# Start training
trainer.train()

# Example usage of the fine-tuned model
def generate_control(business_process, risk):
    prompt = f"Business Process: {business_process}
Risk: {risk}
Control:"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs['input_ids'], max_length=256)
    generated_control = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_control

# Test the model
business_process = "Order to Cash"
risk = "Delivery risk"
control = generate_control(business_process, risk)
print(f"Generated Control for '{risk}' in '{business_process}': {control}")
