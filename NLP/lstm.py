To put everything together into a single script, we'll start from data generation, preprocess the text data for LSTM, build an LSTM model, train it on the generated data, and then evaluate its performance by calculating precision, recall, and F1-score. This script assumes you're using TensorFlow and Keras in your Python environment.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Generate synthetic data
data = {
    "Product ID": np.arange(1, 101),
    "Product Name": ["Product " + str(i) for i in range(1, 101)],
    "Product Description": ["Description for product " + str(i) for i in range(1, 101)],
    "Flag": np.random.choice(["Yes", "No"], 100)
}
df = pd.DataFrame(data)

# Preprocess data
df['Text'] = df['Product Name'] + " " + df['Product Description']
text = df['Text'].values
labels = np.where(df['Flag'] == "Yes", 1, 0)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(text)
sequences = tokenizer.texts_to_sequences(text)
max_len = max(len(x) for x in sequences)
vocab_size = len(tokenizer.word_index) + 1

X = pad_sequences(sequences, maxlen=max_len)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Build LSTM model
model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=max_len))
model.add(LSTM(64))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Predict and evaluate
predicted_probs = model.predict(X_test)
predicted_labels = (predicted_probs > 0.5).astype(int)

precision = precision_score(y_test, predicted_labels)
recall = recall_score(y_test, predicted_labels)
f1 = f1_score(y_test, predicted_labels)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")
```

This script covers:
- Synthetic data generation.
- Preprocessing (combining text fields, tokenizing, padding).
- Building a simple LSTM model suitable for binary classification.
- Training the model on the synthetic dataset.
- Predicting on the test set.
- Evaluating the model's performance using precision, recall, and F1-score.

Before running this script, ensure you have TensorFlow and Keras installed in your Python environment (`pip install tensorflow`). Note that model training parameters such as the number of epochs, batch size, and model architecture (e.g., the number of LSTM units) are set for demonstration purposes and may require adjustment for optimal performance on your specific dataset.
