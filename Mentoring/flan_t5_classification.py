
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW
from sklearn.metrics import f1_score

# Create the dataset
data = {
    'product_id': [1, 2, 3, 4, 5],
    'product_name': ['Product A', 'Product B', 'Product C', 'Product D', 'Product E'],
    'product_desc': [
        'A high-quality product for daily use.',
        'An affordable product with great features.',
        'A luxury product for special occasions.',
        'A reliable product with a long lifespan.',
        'An innovative product with cutting-edge technology.'
    ],
    'applicable': ['y', 'n', 'y', 'y', 'n']
}

df = pd.DataFrame(data)

# Initialize tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-small')
model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-small')

# Tokenize the input data
def preprocess_data(product_name, product_desc, applicable):
    input_text = f"product_name: {product_name} product_desc: {product_desc}"
    target_text = applicable
    return input_text, target_text

# Prepare the data
inputs = []
targets = []
for index, row in df.iterrows():
    input_text, target_text = preprocess_data(row['product_name'], row['product_desc'], row['applicable'])
    inputs.append(input_text)
    targets.append(target_text)

# Tokenize inputs and targets
input_encodings = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
target_encodings = tokenizer(targets, padding=True, truncation=True, return_tensors="pt")

# Convert target_encodings to input_ids for the model
labels = target_encodings.input_ids
labels[labels == tokenizer.pad_token_id] = -100  # Replace padding token id's with -100 to ignore them during loss calculation

class ProductDataset(Dataset):
    def __init__(self, input_encodings, labels):
        self.input_encodings = input_encodings
        self.labels = labels

    def __len__(self):
        return len(self.input_encodings.input_ids)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.input_encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

# Create dataset and dataloader
dataset = ProductDataset(input_encodings, labels)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Define optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop
model.train()
for epoch in range(3):  # Train for 3 epochs
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch}, Loss: {loss.item()}")

# Prepare the model for evaluation
model.eval()

# Function to classify and get probability
def classify_product_with_prob(product_name, product_desc):
    input_text = f"product_name: {product_name} product_desc: {product_desc}"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    outputs = model.generate(input_ids)
    logits = model(input_ids).logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return prediction, probs

# Evaluate on the same dataset for simplicity
true_labels = []
pred_labels = []
for index, row in df.iterrows():
    true_label = row['applicable']
    prediction, prob = classify_product_with_prob(row['product_name'], row['product_desc'])
    true_labels.append(true_label)
    pred_labels.append(prediction)
    print(f"Product: {row['product_name']} | Prediction: {prediction} | True: {true_label} | Probabilities: {prob}")

# Calculate F1 score
f1 = f1_score(true_labels, pred_labels, pos_label='y', average='binary')
print(f"F1 Score: {f1}")
