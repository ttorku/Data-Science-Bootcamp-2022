import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
import torch
import joblib
import os
import time

# Initialize BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Define the specific class names
class_names = ['HR', 'Finance', 'IT', 'Legal', 'Marketing', 'Operations', 'Sales', 'R&D', 'Customer_Support', 'Engineering', 'Compliance', 'Procurement']

def extract_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()  # Using the [CLS] token's embedding

def save_embeddings(embeddings, filename='embeddings.pkl'):
    joblib.dump(embeddings, filename)

def load_embeddings(filename='embeddings.pkl'):
    return joblib.load(filename)

def train_model(X, y, model_type='logistic_regression'):
    if model_type == 'logistic_regression':
        model = LogisticRegression()
        param_grid = {'C': [0.01, 0.1, 1, 10]}
    elif model_type == 'random_forest':
        model = RandomForestClassifier()
        param_grid = {'n_estimators': [100, 200, 500], 'max_depth': [10, 20, 30]}
    elif model_type == 'xgboost':
        model = XGBClassifier()
        param_grid = {'n_estimators': [100, 200], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1, 0.2]}
    else:
        raise ValueError("Invalid model_type. Choose from 'logistic_regression', 'random_forest', 'xgboost'.")

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X, y)
    print(f"Best parameters for {model_type}: {grid_search.best_params_}")
    return grid_search.best_estimator_

def train_and_save_embeddings(df, model_type='logistic_regression', embeddings_filename='embeddings.pkl', model_filename='model.pkl', metrics_filename='metrics.txt'):
    start_time = time.time()
    
    # Extract embeddings
    df['embeddings'] = df.apply(lambda x: extract_embeddings(f"{x['policy_title']} {x['policy_summary']}"), axis=1)
    X = np.vstack(df['embeddings'].values)
    y = df[class_names].values
    
    save_embeddings(X, embeddings_filename)
    
    # Train model
    model = train_model(X, y, model_type)
    joblib.dump(model, model_filename)
    
    # Calculate training time
    training_time = time.time() - start_time
    
    # Evaluate model
    y_pred = model.predict(X)
    metrics = {}
    metrics['accuracy'] = accuracy_score(y, y_pred)
    metrics['precision'] = precision_score(y, y_pred, average='weighted')
    metrics['recall'] = recall_score(y, y_pred, average='weighted')
    metrics['f1_score'] = f1_score(y, y_pred, average='weighted')
    metrics['training_time'] = training_time
    
    # Save metrics
    with open(metrics_filename, 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    
    return model

def inference(df, embeddings_filename='embeddings.pkl', model_filename='model.pkl', threshold=0.5, metrics_filename='metrics.txt'):
    start_time = time.time()
    
    X = load_embeddings(embeddings_filename)
    model = joblib.load(model_filename)
    
    # Predict
    y_pred_prob = model.predict_proba(X) if hasattr(model, "predict_proba") else model.predict(X)
    y_pred = (y_pred_prob >= threshold).astype(int)
    
    # Format the output as "Class (y/n (value))"
    output_df = df[['policy_id', 'policy_title', 'policy_summary']].copy()
    for i, class_name in enumerate(class_names):
        y_n = 'y' if y_pred[:, i] else 'n'
        output_df[class_name] = [f"{y_n} ({prob:.2f})" for prob in y_pred_prob[:, i]]
    
    # Calculate inference time
    inference_time = time.time() - start_time
    
    # Save inference time
    with open(metrics_filename, 'a') as f:
        f.write(f"inference_time: {inference_time}\n")
    
    return output_df

# Example usage
if __name__ == "__main__":
    # Load your dataset
    df = pd.read_csv("policies.csv")  # Ensure your CSV contains the appropriate class names as columns

    # Training
    model = train_and_save_embeddings(df, model_type='xgboost', embeddings_filename='embeddings.pkl', model_filename='xgboost_model.pkl', metrics_filename='xgboost_metrics.txt')

    # Inference
    inference_df = inference(df, embeddings_filename='embeddings.pkl', model_filename='xgboost_model.pkl', threshold=0.5, metrics_filename='xgboost_metrics.txt')
    print(inference_df)
