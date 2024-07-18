import pandas as pd
from datasets import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling

# Example data
data = {
    "process": ["Process A", "Process B"],
    "risk": ["Risk A", "Risk B"],
    "control": ["Control A", "Control B"]
}
df = pd.DataFrame(data)

# Convert DataFrame to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Load tokenizer and model
model_name_or_path = "gpt2-medium"
tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
model = GPT2LMHeadModel.from_pretrained(model_name_or_path)

# Define a function to preprocess the data
def preprocess_function(examples):
    inputs = [f"Process: {proc}\nRisk: {risk}" for proc, risk in zip(examples['process'], examples['risk'])]
    targets = examples['control']
    full_text = [inp + "\nControl: " + tgt for inp, tgt in zip(inputs, targets)]
    model_inputs = tokenizer(full_text, max_length=512, truncation=True, padding="max_length")
    return model_inputs

# Tokenize the dataset
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Remove unnecessary columns
tokenized_datasets = tokenized_datasets.remove_columns(['process', 'risk', 'control'])

# Set training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=3,
    fp16=True  # Use mixed precision for faster training
)

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets,  # In practice, use a separate validation set
    data_collator=data_collator,
    tokenizer=tokenizer
)

# Train the model
trainer.train()

# Save the model
trainer.save_model("./fine-tuned-gpt2-medium")

# Evaluate the model
results = trainer.evaluate()
print(results)

# Example inference
def generate_control(process, risk):
    input_text = f"Process: {process}\nRisk: {risk}\nControl:"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding="max_length", max_length=512).to("cuda")
    outputs = model.generate(**inputs, max_length=512)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

process = "Example Process"
risk = "Example Risk"
print(generate_control(process, risk))
