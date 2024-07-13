# llama_control_des.py

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq, pipeline
from datasets import Dataset
import pandas as pd

def prepare_training_data():
    """
    Prepare the training data as pairs of input and output.
    """
    train_data = [
        {"input": "Data backup + Data loss due to hardware failure", "output": "Implement regular data backups and store them in multiple locations."},
        # Add more data pairs here
    ]

    # Convert the data to a Hugging Face Dataset
    train_dataset = Dataset.from_pandas(pd.DataFrame(train_data))

    return train_dataset

def preprocess_function(examples, tokenizer):
    """
    Preprocess the data for training.
    """
    inputs = examples['input']
    outputs = examples['output']
    model_inputs = tokenizer(inputs, max_length=512, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(outputs, max_length=512, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def fine_tune_model():
    """
    Fine-tune the model using the prepared dataset.
    """
    model_name_or_path = "TheBloke/Llama-2-13b-Chat-GPTQ"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

    # Prepare the training data
    train_dataset = prepare_training_data()
    train_dataset = train_dataset.map(lambda examples: preprocess_function(examples, tokenizer), batched=True)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=3,
        save_steps=500,
    )

    # Define data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # Fine-tune the model
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained("./fine-tuned-model")
    tokenizer.save_pretrained("./fine-tuned-model")

def load_fine_tuned_model(model_path="./fine-tuned-model"):
    """
    Load the fine-tuned model and tokenizer from the specified path.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    return tokenizer, model

def generate_control_description(process, risk, model_path="./fine-tuned-model"):
    """
    Generate a control description given a process and associated risk.
    
    Parameters:
    - process: The process description.
    - risk: The associated risk description.
    - model_path: Path to the fine-tuned model directory.
    
    Returns:
    - control_description: The generated control description.
    """
    tokenizer, model = load_fine_tuned_model(model_path)
    
    # Create a text generation pipeline
    text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
    
    # Prepare the input prompt
    prompt = f"Given the process '{process}' which has the associated risk '{risk}', describe appropriate control measures."
    
    # Generate the control description
    result = text_generator(prompt, max_length=150, num_return_sequences=1)
    control_description = result[0]['generated_text']
    
    return control_description

if __name__ == "__main__":
    # Fine-tune the model (run this only once or when you have new training data)
    fine_tune_model()

    # Example usage
    process = "Data backup"
    risk = "Data loss due to hardware failure"
    control_description = generate_control_description(process, risk)
    print(control_description)
