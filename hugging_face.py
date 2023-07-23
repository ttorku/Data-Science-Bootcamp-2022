#hugging face
import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.metrics import f1_score

# Example data
obligations = [
    "Ensure data privacy and protection.",
    "Implement regular security audits.",
    "Train employees on security best practices."
]

controls = [
    "Encryption of sensitive data.",
    "Penetration testing of systems.",
    "Security awareness training."
]

labels = [1, 0, 1]  # Example labels (0 or 1) indicating if the control is applicable to the obligation

# Combine obligations and controls into a single list
data = obligations + controls

# Tokenize the text data using DistilBERT tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
encoded_data = tokenizer(data, padding=True, truncation=True, return_tensors='tf')

# Convert labels to TensorFlow tensors
labels = tf.convert_to_tensor(labels)

# Split data into train and test sets
train_size = int(0.8 * len(data))
train_data = {key: value[:train_size] for key, value in encoded_data.items()}
test_data = {key: value[train_size:] for key, value in encoded_data.items()}

# Define the model architecture (DistilBERT for sequence classification)
model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    compute_metrics=lambda p: {'f1_score': f1_score(p.label_ids, p.predictions.argmax(axis=1))}
)

# Train the model
trainer.train()

# Evaluate the model on test data
results = trainer.evaluate()

# Print the F1 score
print("F1 Score:", results['eval_f1_score'])

