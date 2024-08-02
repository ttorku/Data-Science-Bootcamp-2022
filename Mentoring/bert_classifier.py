import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

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
        labels = torch.tensor(self.df.iloc[idx][['dept_A', 'dept_B', 'dept_C']].values, dtype=torch.float)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

# Load data
df = pd.read_csv('process_data.csv')  # assuming a CSV file with process_name, process_desc, dept_A, dept_B, dept_C

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
model = BertForSequenceClassification.from_pretrained('bert-large-uncased', num_labels=3)

# Step 3: Train the Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

for epoch in range(3):  # number of epochs
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    val_preds = []
    val_labels = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            val_preds.extend(torch.sigmoid(logits).cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_preds = (np.array(val_preds) > 0.5).astype(int)
    val_labels = np.array(val_labels)
    
    accuracy = accuracy_score(val_labels, val_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(val_labels, val_preds, average='micro')
    
    print(f'Epoch {epoch + 1} | Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f}')

# Step 4: Make Predictions
def predict(model, process_name, process_desc, tokenizer, max_len=128):
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
        outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    preds = torch.sigmoid(logits).cpu().numpy()
    return (preds > 0.5).astype(int)

# Example prediction
new_process_name = "New Process Name"
new_process_desc = "Description of the new process"
prediction = predict(model, new_process_name, new_process_desc, tokenizer)
print(f"Prediction for Dept A: {'Yes' if prediction[0][0] == 1 else 'No'}")
print(f"Prediction for Dept B: {'Yes' if prediction[0][1] == 1 else 'No'}")
print(f"Prediction for Dept C: {'Yes' if prediction[0][2] == 1 else 'No'}")
