
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# Load your dataset
data = pd.read_csv('your_dataset.csv')

# Separate the features (X) from the target variable (y)
X = data.drop(columns=['Disclosure Flag'])
y = data['Disclosure Flag']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Calculate scale_pos_weight
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

# Define the resampling strategy
over_sampler = RandomOverSampler(sampling_strategy=0.5)  # 33% of the majority class
under_sampler = RandomUnderSampler(sampling_strategy=1.0)  # Equal number of samples in each class

# Create the resampling pipeline
resample_pipeline = Pipeline([
    ('over', over_sampler),
    ('under', under_sampler),
])

# Apply the resampling strategy to the training data
X_resampled, y_resampled = resample_pipeline.fit_resample(X_train, y_train)

# Initialize the XGBoost classifier with scale_pos_weight
xgb_classifier = XGBClassifier(scale_pos_weight=scale_pos_weight)

# Train the XGBoost classifier on the resampled data
xgb_classifier.fit(X_resampled, y_resampled)

# Make predictions on the test set
y_pred = xgb_classifier.predict(X_test)

# Print the classification report
print(classification_report(y_test, y_pred))
