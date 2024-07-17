import pandas as pd
from datasets import load_dataset, Dataset
from transformers import BloomForCausalLM, BloomTokenizerFast, Trainer, TrainingArguments, DataCollatorForSeq2Seq
import torch

# Load and prepare the dataset
data = {
    "process": ["Process A", "Process B"],
    "risk": ["Risk A", "Risk B"],
    "control": ["Control A", "Control B"]
}
df = pd.DataFrame(data)

# Convert DataFrame to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Define a function to preprocess the data
def preprocess_function(examples):
    inputs = [f"Process: {proc}\nRisk: {risk}" for proc, risk in zip(examples['process'], examples['risk'])]
    targets = examples['control']
    model_inputs = tokenizer(inputs, max_length=512, truncation=True)
    labels = tokenizer(targets, max_length=512, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Load tokenizer and model
model_name_or_path = "bigscience/bloom-1b7"
tokenizer = BloomTokenizerFast.from_pretrained(model_name_or_path)
model = BloomForCausalLM.from_pretrained(model_name_or_path)

# Tokenize the dataset
tokenized_datasets = dataset.map(preprocess_function, batched=True)

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
    predict_with_generate=True,
    fp16=True  # Use mixed precision for faster training
)

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

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
trainer.save_model("./fine-tuned-bloom")

# Evaluate the model
results = trainer.evaluate()
print(results)

# Example inference
def generate_control(process, risk):
    input_text = f"Process: {process}\nRisk: {risk}"
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

process = "Example Process"
risk = "Example Risk"
print(generate_control(process, risk))
