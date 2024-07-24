import pandas as pd
from datasets import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers import IntervalStrategy
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

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

# Set EOS token as PAD token
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained(model_name_or_path)

# Define a function to preprocess the data
def preprocess_function(examples):
    inputs = [f"Process: {proc}\nRisk: {risk} <sep>" for proc, risk in zip(examples['process'], examples['risk'])]
    targets = examples['control']
    full_text = [inp + " " + tgt for inp, tgt in zip(inputs, targets)]
    model_inputs = tokenizer(full_text, max_length=512, truncation=True, padding="max_length")
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs

# Tokenize the dataset
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Remove unnecessary columns
tokenized_datasets = tokenized_datasets.remove_columns(['process', 'risk', 'control'])

# Set training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy=IntervalStrategy.EPOCH,
    logging_dir='./logs',
    logging_steps=10,  # Log every 10 steps
    save_steps=10,  # Save every 10 steps
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=3,
    fp16=True  # Use mixed precision for faster training
)

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

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

# Create a pipeline for text generation
text_generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0  # Use GPU if available, -1 for CPU
)

# Example inference with temperature
def generate_control(process, risk, temperature=0.7, max_length=512, num_return_sequences=1):
    input_text = f"Process: {process}\nRisk: {risk}\nControl:"
    generated_texts = text_generator(
        input_text,
        max_length=max_length,
        temperature=temperature,
        num_return_sequences=num_return_sequences,
        pad_token_id=tokenizer.eos_token_id
    )
    generated_control = [text['generated_text'].split('Control:')[-1].strip() for text in generated_texts]
    return generated_control[0]





# Example inference
def generate_control(process, risk):
    input_text = f"Process: {process}\nRisk: {risk} <sep>"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding="max_length", max_length=512).to("cuda")
    outputs = model.generate(**inputs, max_length=512, pad_token_id=tokenizer.eos_token_id)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text.split('<sep>')[-1].strip()

process = "Example Process"
risk = "Example Risk"
print(generate_control(process, risk))

