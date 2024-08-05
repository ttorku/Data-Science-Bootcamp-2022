import pandas as pd
import torch
import numpy as np
import time
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn

# Custom BERT model for multi-label classification
class BertForMultiLabelClassification(nn.Module):
    def __init__(self, model_name, num_labels):
        super(BertForMultiLabelClassification, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        nn.init.xavier_uniform_(self.classifier.weight)  # Initialize the classifier layer weights

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            pos_weight = torch.ones(labels.size(1)).to(device)  # Adjust pos_weight for each class if needed
            loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            loss = loss_fct(logits, labels)

        return loss, logits

# Step 1: Prepare Data
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

# Load data
df = pd.read_csv('process_data.csv')  # assuming a CSV file with process_name, process_desc, and 13 department columns

# Split data into train and test sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

# Create DataLoader
train_dataset = ProcessDataset(train_df, tokenizer, max_len=128)
val_dataset = ProcessDataset(val_df, tokenizer, max_len=128)
train_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=8)
val_loader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=8)

# Step 2: Build the Model
num_labels = 13  # Number of labels for multi-label classification
model = BertForMultiLabelClassification('bert-large-uncased', num_labels)

# Step 3: Train the Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Lower learning rate
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# Gradient clipping value
max_grad_norm = 1.0

# Start timing the training process
start_time = time.time()

for epoch in range(3):  # number of epochs
    model.train()
    total_loss = 0
    for step, batch in enumerate(train_loader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        loss, logits = model(input_ids, attention_mask, labels=labels)
        
        # Check for NaN loss
        if torch.isnan(loss).any():
            print(f"NaN loss encountered at epoch {epoch+1}, step {step}")
            continue

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()
        total_loss += loss.item()
        if step % 50 == 0 and step != 0:
            print(f"Epoch {epoch+1}, Step {step}, Loss: {loss.item()}")

    avg_train_loss = total_loss / len(train_loader)

    # Evaluation
    model.eval()
    val_preds = []
    val_labels = []
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            loss, logits = model(input_ids, attention_mask, labels=labels)
            val_loss += loss.item()
            val_preds.extend(torch.sigmoid(logits).cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    
    # Adjust threshold here
    threshold = 0.3
    val_preds = (np.array(val_preds) > threshold).astype(int)
    val_labels = np.array(val_labels)
    
    accuracy = accuracy_score(val_labels, val_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(val_labels, val_preds, average='samples', zero_division=1)
    
    print(f'Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f}')

# End timing the training process
end_time = time.time()
total_training_time = end_time - start_time
print(f"Total Training Time: {total_training_time:.2f} seconds")

# Save the trained model
model_save_path = 'bert_multilabel_model.pt'
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# Function to load the model
def load_model(model_path, model_name='bert-large-uncased', num_labels=13):
    model = BertForMultiLabelClassification(model_name, num_labels)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    return model

# Step 4: Make Predictions
def predict(model, df, tokenizer, max_len=128, threshold=0.3):
    results = []

    for idx, row in df.iterrows():
        process_id = row['Process ID']
        process_name = row['Process Name']
        process_desc = row['Process Description']

        inputs = tokenizer.encode_plus(
            process_name, 
            process_desc, 
            add_special_tokens=True, 
            max_length=max_len,
            padding='max_length', 
            truncation=True, 
            return_tensors='pt'
        )
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        with torch.no_grad():
            _, logits = model(input_ids, attention_mask=attention_mask)
        
        preds = torch.sigmoid(logits).cpu().numpy()
        
        results.append({
            'Process ID': process_id,
            'Process Name': process_name,
            'Process Description': process_desc,
            'Dept 1 Applicability': f"Yes ({preds[0][0]:.2f})" if preds[0][0] > threshold else f"No ({preds[0][0]:.2f})",
                        'Dept 2 Applicability': f"Yes ({preds[0][1]:.2f})" if preds[0][1] > threshold else f"No ({preds[0][1]:.2f})",
            'Dept 3 Applicability': f"Yes ({preds[0][2]:.2f})" if preds[0][2] > threshold else f"No ({preds[0][2]:.2f})",
            'Dept 4 Applicability': f"Yes ({preds[0][3]:.2f})" if preds[0][3] > threshold else f"No ({preds[0][3]:.2f})",
            'Dept 5 Applicability': f"Yes ({preds[0][4]:.2f})" if preds[0][4] > threshold else f"No ({preds[0][4]:.2f})",
            'Dept 6 Applicability': f"Yes ({preds[0][5]:.2f})" if preds[0][5] > threshold else f"No ({preds[0][5]:.2f})",
            'Dept 7 Applicability': f"Yes ({preds[0][6]:.2f})" if preds[0][6] > threshold else f"No ({preds[0][6]:.2f})",
            'Dept 8 Applicability': f"Yes ({preds[0][7]:.2f})" if preds[0][7] > threshold else f"No ({preds[0][7]:.2f})",
            'Dept 9 Applicability': f"Yes ({preds[0][8]:.2f})" if preds[0][8] > threshold else f"No ({preds[0][8]:.2f})",
            'Dept 10 Applicability': f"Yes ({preds[0][9]:.2f})" if preds[0][9] > threshold else f"No ({preds[0][9]:.2f})",
            'Dept 11 Applicability': f"Yes ({preds[0][10]:.2f})" if preds[0][10] > threshold else f"No ({preds[0][10]:.2f})",
            'Dept 12 Applicability': f"Yes ({preds[0][11]:.2f})" if preds[0][11] > threshold else f"No ({preds[0][11]:.2f})",
            'Dept 13 Applicability': f"Yes ({preds[0][12]:.2f})" if preds[0][12] > threshold else f"No ({preds[0][12]:.2f})",
        })

    return pd.DataFrame(results)

# Load the model for inference
loaded_model = load_model(model_save_path)

# Example usage
test_df = pd.DataFrame({
    'Process ID': [1, 2],
    'Process Name': ["New Process 1", "New Process 2"],
    'Process Description': ["Description of the first new process", "Description of the second new process"]
})

prediction_df = predict(loaded_model, test_df, tokenizer)
print(prediction_df)

