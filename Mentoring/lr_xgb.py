import os
import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import joblib

# Define your custom dataset class
class ProcessDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        process_name = self.df.iloc[idx]['process_name']
        process_desc = self.df.iloc[idx]['process_desc']
        inputs = self.tokenizer.encode_plus(
            process_name,
            process_desc,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

# Load the data
df = pd.read_csv('process_data.csv')
df = df.fillna('')

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
bert_model = BertModel.from_pretrained('bert-large-uncased')

# Set up the DataLoader
test_df = pd.DataFrame({
    'Process ID': [1, 2],
    'Process Name': ["New Process 1", "New Process 2"],
    'Process Description': ["Description of the first new process", "Description of the second new process"]
})

test_dataset = ProcessDataset(test_df, tokenizer, max_len=128)
test_loader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=8)

# Extract embeddings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bert_model.to(device)

def extract_embeddings(dataloader):
    bert_model.eval()
    embeddings = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs[1]
            embeddings.append(pooled_output.cpu().numpy())
    embeddings = np.vstack(embeddings)
    return embeddings

test_embeddings = extract_embeddings(test_loader)

# Load your trained model
model_filename = 'model.pkl'
model = joblib.load(model_filename)

# Get predicted probabilities for each department
test_probs = model.predict_proba(test_embeddings)

# Define actual department names in a list
department_names = ['HR', 'Finance', 'IT', 'Marketing', 'Sales', 'Support', 'Operations', 'Legal', 'R&D', 'Product', 'Engineering', 'Customer Service', 'Admin']

# Debugging: Print the structure of test_probs
print("test_probs structure:", len(test_probs), [arr.shape for arr in test_probs])

results = []
for idx, row in test_df.iterrows():
    process_id = row['Process ID']
    process_name = row['Process Name']
    process_desc = row['Process Description']
    result = {
        'Process ID': process_id,
        'Process Name': process_name,
        'Process Description': process_desc,
    }
    # Iterate over the department names
    for i, dept_name in enumerate(department_names):
        # Debugging: Print the current department and index
        print(f"Processing {dept_name} at index {i} for sample {idx}")
        
        try:
            prob = test_probs[i][idx] if isinstance(test_probs[i], np.ndarray) else test_probs[i].item()
        except KeyError as e:
            print(f"KeyError: {e} for department {dept_name} at index {i}")
            continue

        applicability = "Yes" if prob > 0.5 else "No"
        result[f'{dept_name} Applicability'] = f"{applicability} ({prob:.2f})"
    results.append(result)

# Convert results to a DataFrame
results_df = pd.DataFrame(results)
print(results_df)
