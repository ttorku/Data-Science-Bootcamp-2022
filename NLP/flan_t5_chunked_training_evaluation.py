
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
from nltk.translate.bleu_score import sentence_bleu

class ControlsDataset(Dataset):
    def __init__(self, tokenizer, data, max_length=512):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_ids = self.tokenizer(item['input'], max_length=self.max_length, truncation=True, return_tensors='pt').input_ids
        labels = self.tokenizer(item['output'], max_length=self.max_length, truncation=True, return_tensors='pt').input_ids
        return {'input_ids': input_ids.squeeze(), 'labels': labels.squeeze()}

def train_model(data, model, tokenizer, device, epochs=3):
    model.train()
    optimizer = AdamW(model.parameters(), lr=5e-5)
    dataloader = DataLoader(ControlsDataset(tokenizer, data), batch_size=2, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader)}")

def evaluate_model(model, tokenizer, test_data, device):
    model.eval()
    predictions, actuals = [], []
    for item in test_data:
        input_ids = tokenizer(item['input'], return_tensors="pt", max_length=512).input_ids.to(device)
        labels = tokenizer(item['output'], return_tensors="pt", max_length=512).input_ids.to(device)

        with torch.no_grad():
            generated_ids = model.generate(input_ids, max_length=512)

        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        actual_text = tokenizer.decode(labels.squeeze(), skip_special_tokens=True)

        predictions.append(generated_text)
        actuals.append(actual_text)

    precision = precision_score(actuals, predictions, average='macro')
    recall = recall_score(actuals, predictions, average='macro')
    f1 = f1_score(actuals, predictions, average='macro')
    bleu = sentence_bleu([actuals], predictions)

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1}")
    print(f"BLEU Score: {bleu}")

# Dummy data setup (to be replaced with actual data)
data_chunk1 = [
    {"input": "Process: Employee Onboarding, Risk: Misplacement of confidential information", "output": "Why: To secure sensitive data. Where: Digital onboarding system. When: During the initial setup."},
]
data_chunk2 = [
    {"input": "Process: Employee Onboarding, Risk: Misplacement of confidential information", "output": "How: By implementing encrypted storage. What: Encryption of personal data. Who: IT security team."},
]

# Load tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-base')
model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-base')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Train and evaluate first chunk
train_model(data_chunk1, model, tokenizer, device)
evaluate_model(model, tokenizer, data_chunk1, device)

# Optionally reset model or continue training on second chunk
train_model(data_chunk2, model, tokenizer, device)
evaluate_model(model, tokenizer, data_chunk2, device)
