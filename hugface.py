import numpy as np
from transformers import BertTokenizer, TFBertModel

# Example obligations and controls
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

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertModel.from_pretrained('bert-base-uncased')

# Tokenize obligations and controls
obligation_tokens = tokenizer(obligations, padding=True, truncation=True, return_tensors='tf')
control_tokens = tokenizer(controls, padding=True, truncation=True, return_tensors='tf')

# Get BERT embeddings for obligations and controls
obligation_embeddings = model(obligation_tokens.input_ids).pooler_output
control_embeddings = model(control_tokens.input_ids).pooler_output

# Calculate cosine similarity between embeddings
def cosine_similarity(x, y):
    dot_product = np.sum(x * y)
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    similarity = dot_product / (norm_x * norm_y)
    return similarity

# Calculate similarity scores for each obligation-control pair
similarity_scores = []
for ob_emb, ctrl_emb in zip(obligation_embeddings, control_embeddings):
    ob_emb = ob_emb.numpy()
    ctrl_emb = ctrl_emb.numpy()
    similarity_score = cosine_similarity(ob_emb, ctrl_emb)
    similarity_scores.append(similarity_score)

print("Similarity Scores:")
print(similarity_scores)

true_labels = [1, 0, 1]  # True labels (0 or 1) indicating if the control is applicable to the obligation
true_labels = [1, 0, 1]  # True labels (0 or 1) indicating if the control is applicable to the obligation

# Threshold for deciding if control is applicable or not based on similarity score
threshold = 0.5

# Calculate predicted labels based on the similarity scores and the threshold
predicted_labels = [1 if score >= threshold else 0 for score in similarity_scores]

# Calculate true positives (TP), false positives (FP), true negatives (TN), and false negatives (FN)
TP = sum([1 for true, pred in zip(true_labels, predicted_labels) if true == 1 and pred == 1])
FP = sum([1 for true, pred in zip(true_labels, predicted_labels) if true == 0 and pred == 1])
TN = sum([1 for true, pred in zip(true_labels, predicted_labels) if true == 0 and pred == 0])
FN = sum([1 for true, pred in zip(true_labels, predicted_labels) if true == 1 and pred == 0])

# Calculate precision
precision = TP / (TP + FP) if (TP + FP) != 0 else 0.0

# Calculate recall (or sensitivity)
recall = TP / (TP + FN) if (TP + FN) != 0 else 0.0

# Calculate F1 score
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0.0

# Calculate accuracy
accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else 0.0

# Print the evaluation metrics
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)
print("Accuracy:", accuracy)
print("True Positives (TP):", TP)
print("False Positives (FP):", FP)
print("True Negatives (TN):", TN)
print("False Negatives (FN):", FN)

#TransformerXL
import numpy as np
from transformers import TransfoXLTokenizer, TFTransfoXLModel

# Example obligations and controls
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

# Load TransformerXL tokenizer and model
tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')
model = TFTransfoXLModel.from_pretrained('transfo-xl-wt103')

# Tokenize obligations and controls
obligation_tokens = tokenizer(obligations, padding=True, truncation=True, return_tensors='tf')
control_tokens = tokenizer(controls, padding=True, truncation=True, return_tensors='tf')

# Get TransformerXL embeddings for obligations and controls
obligation_embeddings = model(obligation_tokens.input_ids).last_hidden_state[:, -1]
control_embeddings = model(control_tokens.input_ids).last_hidden_state[:, -1]

# Calculate cosine similarity between embeddings
def cosine_similarity(x, y):
    dot_product = np.sum(x * y)
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    similarity = dot_product / (norm_x * norm_y)
    return similarity

# Calculate similarity scores for each obligation-control pair
similarity_scores = []
for ob_emb, ctrl_emb in zip(obligation_embeddings, control_embeddings):
    ob_emb = ob_emb.numpy()
    ctrl_emb = ctrl_emb.numpy()
    similarity_score = cosine_similarity(ob_emb, ctrl_emb)
    similarity_scores.append(similarity_score)

print("Similarity Scores:")
print(similarity_scores)




##XLNet

import numpy as np
from transformers import XLNetTokenizer, TFXLNetModel

# Example obligations and controls
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

# Load XLNet tokenizer and model
tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
model = TFXLNetModel.from_pretrained('xlnet-base-cased')

# Tokenize obligations and controls
obligation_tokens = tokenizer(obligations, padding=True, truncation=True, return_tensors='tf')
control_tokens = tokenizer(controls, padding=True, truncation=True, return_tensors='tf')

# Get XLNet embeddings for obligations and controls
obligation_embeddings = model(obligation_tokens.input_ids).last_hidden_state[:, -1]
control_embeddings = model(control_tokens.input_ids).last_hidden_state[:, -1]

# Calculate cosine similarity between embeddings
def cosine_similarity(x, y):
    dot_product = np.sum(x * y)
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    similarity = dot_product / (norm_x * norm_y)
    return similarity

# Calculate similarity scores for each obligation-control pair
similarity_scores = []
for ob_emb, ctrl_emb in zip(obligation_embeddings, control_embeddings):
    ob_emb = ob_emb.numpy()
    ctrl_emb = ctrl_emb.numpy()
    similarity_score = cosine_similarity(ob_emb, ctrl_emb)
    similarity_scores.append(similarity_score)

print("Similarity Scores:")
print(similarity_scores)






