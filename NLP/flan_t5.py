from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

model_name = "google/flan-t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
model.eval()  # Put model in evaluation mode


def get_classification_probabilities(text):
    # Prepend the classification prompt
    input_text = "classify: " + text

    # Encode the text input to tensor
    inputs = tokenizer(input_text, return_tensors="pt").input_ids

    # Get the output logits from the model
    outputs = model(inputs, return_dict=True)

    # Get the logits of the last token in the sequence, which corresponds to the prediction
    last_token_logits = outputs.logits[:, -1, :]

    # Apply softmax to convert logits to probabilities
    probs = torch.softmax(last_token_logits, dim=-1)

    # Map indices to labels
    yes_index = tokenizer.convert_tokens_to_ids('y')
    no_index = tokenizer.convert_tokens_to_ids('n')

    # Extract probabilities of 'y' and 'n'
    yes_prob = probs[:, yes_index].item()
    no_prob = probs[:, no_index].item()

    return {'yes': yes_prob, 'no': no_prob}




import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch.nn.functional as F

# Load model and tokenizer
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.eval()  # Ensure the model is in eval mode

# Example data
data = [{"description": "Great quality product.", "label": "y"},
        {"description": "Did not work as expected.", "label": "n"}]

true_labels = []
predicted_labels = []

for item in data:
    # Tokenize the input text
    input_text = "classify: " + item['description']
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids

    # Generate output logits
    outputs = model(input_ids)
    logits = outputs.logits[:, -1, :]  # We assume last token prediction

    # Apply softmax to convert logits to probabilities
    probabilities = torch.softmax(logits, dim=-1)

    # Extract the index of the highest probability which corresponds to our predicted label
    pred_label_idx = torch.argmax(probabilities, dim=-1).item()
    predicted_label = tokenizer.decode([pred_label_idx])

    # Append to lists
    true_labels.append(item['label'])
    predicted_labels.append(predicted_label)

# Make sure to post-process predicted_labels if necessary to match the true labels (e.g., trimming, casing)
