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
from sklearn.ensemble import VotingClassifier
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

# Function to extract embeddings using BERT
def extract_embeddings(dataloader, model, device):
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs[1]
            embeddings.append(pooled_output.cpu().numpy())
            labels.append(batch['labels'].cpu().numpy())
    embeddings = np.vstack(embeddings)
    labels = np.vstack(labels)
    return embeddings, labels

# Class for training, prediction, and saving models
class EnsembleModelTrainer:
    def __init__(self):
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

    def grid_search_models(self):
        self.train_embeddings, self.train_labels = self.extract_embeddings(self.train_loader)
        self.val_embeddings, self.val_labels = self.extract_embeddings(self.val_loader)
        self.train_labels = self.train_labels.astype(int)
        self.val_labels = self.val_labels.astype(int)

        param_grid_lr = {'C': [0.1, 1, 10]}
        param_grid_rf = {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]}
        param_grid_xgb = {'n_estimators': [100, 200], 'max_depth': [3, 6, 10]}

        lr = GridSearchCV(LogisticRegression(max_iter=1000), param_grid_lr, scoring='f1_samples', cv=3, verbose=1, n_jobs=-1)
        rf = GridSearchCV(RandomForestClassifier(), param_grid_rf, scoring='f1_samples', cv=3, verbose=1, n_jobs=-1)
        xgb = GridSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'), param_grid_xgb, scoring='f1_samples', cv=3, verbose=1, n_jobs=-1)

        lr.fit(self.train_embeddings, self.train_labels)
        rf.fit(self.train_embeddings, self.train_labels)
        xgb.fit(self.train_embeddings, self.train_labels)

        self.best_lr = lr.best_estimator_
        self.best_rf = rf.best_estimator_
        self.best_xgb = xgb.best_estimator_

        print("Best Params LR: ", lr.best_params_)
        print("Best Params RF: ", rf.best_params_)
        print("Best Params XGB: ", xgb.best_params_)

        self.ensemble_model = VotingClassifier(estimators=[
            ('lr', self.best_lr),
            ('rf', self.best_rf),
            ('xgb', self.best_xgb)
        ], voting='soft')

        self.ensemble_model.fit(self.train_embeddings, self.train_labels)
        self.evaluate_model()

    def evaluate_model(self):
        val_preds = self.ensemble_model.predict(self.val_embeddings)
        accuracy = accuracy_score(self.val_labels, val_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(self.val_labels, val_preds, average='samples', zero_division=1)
        self.metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        print(f"Ensemble Model | Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f}")

    def save_model_and_metrics(self):
        save_dir = 'ensemble_model_and_metrics'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model_save_path = os.path.join(save_dir, 'ensemble_model.pkl')
        metrics_save_path = os.path.join(save_dir, 'ensemble_metrics.txt')

        joblib.dump(self.ensemble_model, model_save_path)
        print(f"Ensemble model saved to {model_save_path}")

        with open(metrics_save_path, 'w') as f:
            for key, value in self.metrics.items():
                f.write(f"{key.capitalize()}: {value:.4f}\n")
        print(f"Metrics saved to {metrics_save_path}")

    def predict(self, test_df):
        test_dataset = ProcessDataset(test_df, self.tokenizer, max_len=128)
        test_loader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=8)
        test_embeddings, _ = self.extract_embeddings(test_loader)

        test_preds = self.ensemble_model.predict(test_embeddings)
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
    trainer = EnsembleModelTrainer()
    trainer.load_data('process_data.csv')
    trainer.grid_search_models()
    trainer.save_model_and_metrics()

    # Example prediction
    # test_df = pd.DataFrame
    test_df = pd.DataFrame({
        'Process ID': [1, 2],
        'Process Name': ["New Process 1", "New Process 2"],
        'Process Description': ["Description of the first new process", "Description of the second new process"]
    })
    predictions = trainer.predict(test_df)
    print(predictions)
