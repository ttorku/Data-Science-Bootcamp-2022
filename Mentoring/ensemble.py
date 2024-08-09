import os
import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import joblib

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

# Class for training, prediction, and saving models
class ModelTrainer:
    def __init__(self, model_type):
        self.model_type = model_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        self.bert_model = BertModel.from_pretrained('bert-large-uncased')
        self.bert_model.to(self.device)

    def load_data(self, filepath):
        df = pd.read_csv(filepath)
        df = df.fillna('')
        for column in ['dept_1', 'dept_2', 'dept_3', 'dept_4', 'dept_5', 'dept_6', 'dept_7', 'dept_8', 'dept_9', 'dept_10', 'dept_11', 'dept_12', 'dept_13']:
            assert df[column].var() != 0, f"Column {column} has zero variance in the data"
        self.train_df, self.val_df = train_test_split(df, test_size=0.2, random_state=42)
        self.train_dataset = ProcessDataset(self.train_df, self.tokenizer, max_len=128)
        self.val_dataset = ProcessDataset(self.val_df, self.tokenizer, max_len=128)
        self.train_loader = DataLoader(self.train_dataset, sampler=RandomSampler(self.train_dataset), batch_size=8)
        self.val_loader = DataLoader(self.val_dataset, sampler=SequentialSampler(self.val_dataset), batch_size=8)

    def extract_embeddings(self, dataloader):
        self.bert_model.eval()
        embeddings = []
        labels = []
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
                pooled_output = outputs[1]
                embeddings.append(pooled_output.cpu().numpy())
                labels.append(batch['labels'].cpu().numpy())
        embeddings = np.vstack(embeddings)
        labels = np.vstack(labels)
        return embeddings, labels

    def train_model(self):
        self.train_embeddings, self.train_labels = self.extract_embeddings(self.train_loader)
        self.val_embeddings, self.val_labels = self.extract_embeddings(self.val_loader)
        self.train_labels = self.train_labels.astype(int)
        self.val_labels = self.val_labels.astype(int)

        if self.model_type == 'lr':
            model = LogisticRegression(max_iter=1000)
            param_grid = {'C': [0.1, 1, 10]}
        elif self.model_type == 'rf':
            model = RandomForestClassifier()
            param_grid = {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]}
        elif self.model_type == 'xgb':
            model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
            param_grid = {'n_estimators': [100, 200], 'max_depth': [3, 6, 10]}
        else:
            raise ValueError("Invalid model type. Choose from 'lr', 'rf', 'xgb'.")

        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='f1_samples', cv=3, verbose=1, n_jobs=-1)
        grid_search.fit(self.train_embeddings, self.train_labels)
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.evaluate_model()

    def evaluate_model(self):
        val_preds = self.model.predict(self.val_embeddings)
        accuracy = accuracy_score(self.val_labels, val_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(self.val_labels, val_preds, average='samples', zero_division=1)
        self.metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        print(f"Model: {self.model_type} | Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f}")
        print(f"Best Params: {self.best_params}")

    def save_model_and_metrics(self):
        save_dir = f'{self.model_type}_model_and_metrics'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model_save_path = os.path.join(save_dir, f'{self.model_type}_model.pkl')
        metrics_save_path = os.path.join(save_dir, f'{self.model_type}_metrics.txt')

        joblib.dump(self.model, model_save_path)
        print(f"Model saved to {model_save_path}")

        with open(metrics_save_path, 'w') as f:
            for key, value in self.metrics.items():
                f.write(f"{key.capitalize()}: {value:.4f}\n")
            f.write(f"Best Params: {self.best_params}\n")
        print(f"Metrics saved to {metrics_save_path}")

    def predict(self, test_df):
        test_dataset = ProcessDataset(test_df, self.tokenizer, max_len=128)
        test_loader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=8)
        test_embeddings, _ = self.extract_embeddings(test_loader)

        test_preds = self.model.predict(test_embeddings)
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
            for i in range(13):
                result[f'Dept {i+1} Applicability'] = "Yes" if test_preds[idx, i] == 1 else "No"
            results.append(result)

        return pd.DataFrame(results)

# Usage example:
if __name__ == '__main__':
    model_type = input("Enter model type (lr, rf, xgb): ").strip().lower()
    trainer = ModelTrainer(model_type)
    trainer.load_data('process_data.csv')
    trainer.train_model()
    trainer.save_model_and_metrics()

    # Example prediction
    test_df = pd.DataFrame({
        'Process ID': [1, 2],
        'Process Name': ["New Process 1", "New Process 2"],
        'Process Description': ["Description of the first new process", "Description of the second new process"]
    })
    predictions = trainer.predict(test_df)
    print(predictions)

def predict(self, test_df):
    test_dataset = ProcessDataset(test_df, self.tokenizer, max_len=128)
    test_loader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=8)
    test_embeddings, _ = self.extract_embeddings(test_loader)

    # Get predicted probabilities
    test_probs = self.model.predict_proba(test_embeddings)

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
        for i in range(13):
            prob = test_probs[i][idx][:].item() if isinstance(test_probs[i], np.ndarray) else test_probs[i].item()
            applicability = "Yes" if prob > 0.5 else "No"
            result[f'Dept {i+1} Applicability'] = f"{applicability} ({prob:.2f})"
        results.append(result)

    return pd.DataFrame(results)

