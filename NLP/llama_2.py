import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

# Check for GPU availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the tokenizer and model (replace 'llama2' with the actual model identifier)
tokenizer = AutoTokenizer.from_pretrained('llama2')
model = AutoModelForCausalLM.from_pretrained('llama2').to(device)

# Prepare the dataset (replace 'path_to_your_dataset' with the path to your dataset)
train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path='path_to_your_dataset',
    block_size=128
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=False  # Set to False for causal language modeling (GPT-like models)
)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',           # Output directory for model checkpoints
    overwrite_output_dir=True,        # Overwrite the content of the output directory
    num_train_epochs=3,               # Number of training epochs
    per_device_train_batch_size=4,    # Batch size per device (GPU)
    save_steps=10_000,                # Number of updates steps before saving a checkpoint
    save_total_limit=2,               # Number of total checkpoints to keep
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

# Start training
trainer.train()

# Save the fine-tuned model
model.save_pretrained('./fine_tuned_llama2')
