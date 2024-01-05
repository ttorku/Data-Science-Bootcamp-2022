from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Initialize classifiers
log_reg = LogisticRegression()
mlp = MLPClassifier()
svc = SVC()
decision_tree = DecisionTreeClassifier()
random_forest = RandomForestClassifier()

# Dictionary to hold classifiers
classifiers = {
    "Logistic Regression": log_reg,
    "Neural Network (MLP)": mlp,
    "Support Vector Machine (SVM)": svc,
    "Decision Tree": decision_tree,
    "Random Forest": random_forest
}

# Training and evaluating each classifier
results = {}
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    results[name] = score

results



# Re-preparing the data using only 'Product Description' and 'Flag'

# Extracting features and target
X = vectorizer.transform(uploaded_data['Product Description']).toarray()
y = uploaded_data['Flag']

# Splitting the data into training and testing sets again
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Training and evaluating each classifier again with the updated data
updated_results = {}
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    updated_results[name] = score

updated_results


# Correcting the process to add predictions to the DataFrame
for name, clf in classifiers.items():
    predictions = clf.predict(X_out_of_sample)
    # Inverting the label encoding to get 'Yes' or 'No'
    inverted_predictions = label_encoder.inverse_transform(predictions)
    predictions_df[name + ' Prediction'] = inverted_predictions

predictions_df


# Correcting the summary process for precision, recall, and f1-score

# Summarizing precision, recall, and f1-score for each classifier
summary_corrected = {}
for name, report in evaluation_results.items():
    summary_metrics = {metric: report['weighted avg'][metric] for metric in ['precision', 'recall', 'f1-score']}
    summary_corrected[name] = summary_metrics

summary_corrected




# Importing required libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Load the data
file_path = '/mnt/data/Business_Data (2).csv'
data = pd.read_csv(file_path)

# Preprocessing the data
# Encoding the 'Flag' column
label_encoder = LabelEncoder()
data['Flag'] = label_encoder.fit_transform(data['Flag'])

# Extracting features from 'Product Description'
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['Product Description']).toarray()
y = data['Flag']

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Initialize classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(),
    "Neural Network (MLP)": MLPClassifier(),
    "Support Vector Machine (SVM)": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

# Training and evaluating classifiers
evaluation_results = {}
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    evaluation_results[name] = report

# Summarizing precision, recall, and f1-score for each classifier
summary = {}
for name, report in evaluation_results.items():
    summary_metrics = {metric: report['weighted avg'][metric] for metric in ['precision', 'recall', 'f1-score']}
    summary[name] = summary_metrics

summary


# Creating out-of-sample data for prediction
np.random.seed(1)
out_of_sample_descriptions = [np.random.choice(example_descriptions) for _ in range(20)]
out_of_sample_data = pd.DataFrame({
    'Product Description': out_of_sample_descriptions
})

# Transforming the out-of-sample data for model input
X_out_of_sample = vectorizer.transform(out_of_sample_data['Product Description']).toarray()

# Predicting using each classifier
out_of_sample_predictions = {}
for name, clf in classifiers.items():
    predictions = clf.predict(X_out_of_sample)
    out_of_sample_predictions[name] = label_encoder.inverse_transform(predictions)

out_of_sample_data['Predictions'] = out_of_sample_predictions
out_of_sample_data




