
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW

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

# Custom PyTorch Dataset
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        return {key: torch.tensor(val) for key, val in item.items()}

# Prepare DataLoader
train_dataset = CustomDataset(tokenized_dataset)
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# Set up optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop
model.train()
num_epochs = 3

for epoch in range(num_epochs):
    for batch in train_dataloader:
        optimizer.zero_grad()
        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# Save the fine-tuned model
model.save_pretrained('./fine_tuned_llama2')

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
