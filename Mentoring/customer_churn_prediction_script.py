
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Assuming df is your DataFrame after data cleaning

# Feature Engineering
# Convert categorical variables into dummy variables
categorical_features = ['Contract', 'OnlineSecurity', 'TechSupport', 'StreamingTV']
categorical_transformer = OneHotEncoder(drop='first')

# Preprocessor pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features),
    ])

# Model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('scaler', StandardScaler(with_mean=False)),
    ('classifier', LogisticRegression(solver='liblinear'))
])

# Prepare target variable and features
X = df.drop(['CustomerID', 'Churn', 'TotalCharges'], axis=1)
y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the model
model.fit(X_train, y_train)

# Predicting on the test set
y_pred = model.predict(X_test)

# Evaluating the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

# Model Interpretation
# Getting the feature names after one-hot encoding
feature_names = list(model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features))
feature_names += ['Tenure', 'MonthlyCharges']  # Adding the numerical features

# Displaying the coefficients
coefs = model.named_steps['classifier'].coef_[0]
coefficients = pd.DataFrame(coefs, index=feature_names, columns=['Coefficient'])
print(coefficients)

# Business Insights
# Based on the coefficients, we can interpret the influence of various features on the likelihood of churn.
# For instance, higher monthly charges or lack of online security might increase the likelihood of churn.
