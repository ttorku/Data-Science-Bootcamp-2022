import os
import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier

# Custom Dataset class
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
        labels = torch.tensor(self.df.iloc[idx][['dept_1', 'dept_2', 'dept_3', 'dept_4', 'dept_5', 'dept_6', 'dept_7', 'dept_8', 'dept_9', 'dept_10', 'dept_11', 'dept_12', 'dept_13']].values, dtype=torch.float)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

# Function to extract embeddings using BERT
def extract_embeddings(dataloader, model, device):
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs[1]
            embeddings.append(pooled_output.cpu().numpy())
            labels.append(batch['labels'].cpu().numpy())
    embeddings = np.vstack(embeddings)
    labels = np.vstack(labels)
    return embeddings, labels

# Load data
df = pd.read_csv('process_data.csv')  # assuming a CSV file with process_name, process_desc, and 13 department columns

# Check for NaN or infinite values in the data
print("Checking for NaN values...")
nan_columns = df.columns[df.isnull().any()]
nan_counts = df[nan_columns].isnull().sum()
print(f"Columns with NaN values:\n{nan_counts}")

# Handling NaN values: Fill NaNs with appropriate values or drop rows
df = df.fillna('')  # Example: Filling NaNs with empty strings for text columns

# Ensure no label column has zero variance
for column in ['dept_1', 'dept_2', 'dept_3', 'dept_4', 'dept_5', 'dept_6', 'dept_7', 'dept_8', 'dept_9', 'dept_10', 'dept_11', 'dept_12', 'dept_13']:
    assert df[column].var() != 0, f"Column {column} has zero variance in the data"

# Split data into train and test sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Tokenizer and BERT model
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
bert_model = BertModel.from_pretrained('bert-large-uncased')

# Create DataLoader
train_dataset = ProcessDataset(train_df, tokenizer, max_len=128)
val_dataset = ProcessDataset(val_df, tokenizer, max_len=128)
train_loader = DataLoader(train_dataset, sampler=SequentialSampler(train_dataset), batch_size=8)
val_loader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=8)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bert_model.to(device)

# Extract embeddings
train_embeddings, train_labels = extract_embeddings(train_loader, bert_model, device)
val_embeddings, val_labels = extract_embeddings(val_loader, bert_model, device)

# Flatten the labels
train_labels = train_labels.astype(int)
val_labels = val_labels.astype(int)

# Train ensemble models
logistic = LogisticRegression(max_iter=1000)
rf = RandomForestClassifier(n_estimators=100)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

# Voting Classifier
ensemble_model = VotingClassifier(estimators=[
    ('lr', logistic),
    ('rf', rf),
    ('xgb', xgb)
], voting='soft')

# Fit the ensemble model
ensemble_model.fit(train_embeddings, train_labels)

# Predict and evaluate
val_preds = ensemble_model.predict(val_embeddings)

# Evaluate the ensemble model
accuracy = accuracy_score(val_labels, val_preds)
precision, recall, f1, _ = precision_recall_fscore_support(val_labels, val_preds, average='samples', zero_division=1)

print(f"Ensemble Model | Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f}")

# Save the ensemble model
import joblib
model_save_path = 'ensemble_model.pkl'
joblib.dump(ensemble_model, model_save_path)
print(f"Ensemble model saved to {model_save_path}")

# Function to make predictions using the ensemble model
def predict_ensemble(model_path, df, tokenizer, max_len=128):
    model = joblib.load(model_path)
    dataset = ProcessDataset(df, tokenizer, max_len=max_len)
    dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=8)
    
    embeddings, _ = extract_embeddings(dataloader, bert_model, device)
    preds = model.predict(embeddings)
    
    results = []
    for idx, row in df.iterrows():
        process_id = row['Process ID']
        process_name = row['Process Name']
        process_desc = row['Process Description']
        
        result = {
            'Process ID': process_id,
            'Process Name': process_name,
            'Process Description': process_desc,
        }
        for i in range(13):
            result[f'Dept {i+1} Applicability'] = "Yes" if preds[idx, i] == 1 else "No"
        results.append(result)
    
    return pd.DataFrame(results)

# Example usage
test_df = pd.DataFrame({
    'Process ID': [1, 2],
    'Process Name': ["New Process 1", "New Process 2"],
    'Process Description': ["Description of the first new process", "Description of the second new process"]
})

prediction_df = predict_ensemble(model_save_path, test_df, tokenizer)
print(prediction_df)
