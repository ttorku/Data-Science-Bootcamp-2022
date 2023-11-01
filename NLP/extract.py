import requests
from io import BytesIO

# Read the Excel file
df_updated = pd.read_excel(excel_filename_new)

# Function to download a PDF from a URL and extract the summary
def extract_summary_from_url(url):
    try:
        # Send HTTP GET request to download the PDF
        response = requests.get(url)
        response.raise_for_status()
        
        # Open the PDF from the byte stream
        with BytesIO(response.content) as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page in reader.pages:
                text += page.extract_text()
            # Assuming the summary is located between the words "Summary:" and "Scope:"
            start = text.find("Summary:") + len("Summary:")
            end = text.find("Scope:")
            if start != -1 and end != -1 and start < end:
                return text[start:end].strip()
            else:
                return "Summary not found"
    except requests.HTTPError as http_err:
        return f"HTTP error occurred: {http_err}"
    except Exception as err:
        return f"An error occurred: {err}"

# Extract summaries from the PDFs
df_updated['Summary'] = df_updated['Title Name'].apply(extract_summary_from_url)

# Save the updated DataFrame to an Excel file
updated_excel_filename = '/mnt/data/business_documents_updated.xlsx'
df_updated.to_excel(updated_excel_filename, index=False)

updated_excel_filename, df_updated



import openpyxl
import requests
from io import BytesIO
import pandas as pd
import PyPDF2

def extract_hyperlink(excel_path, sheet_name, cell_address):
    workbook = openpyxl.load_workbook(excel_path)
    sheet = workbook[sheet_name]
    cell = sheet[cell_address]
    if cell.hyperlink:
        return cell.hyperlink.target
    else:
        return None

def extract_summary_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        with BytesIO(response.content) as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page in reader.pages:
                text += page.extract_text()
            start = text.find("Summary:") + len("Summary:")
            end = text.find("Scope:")
            if start != -1 and end != -1 and start < end:
                return text[start:end].strip()
            else:
                return "Summary not found"
    except requests.HTTPError as http_err:
        return f"HTTP error occurred: {http_err}"
    except Exception as err:
        return f"An error occurred: {err}"

def update_excel_with_summaries(excel_path, sheet_name='Sheet1'):
    df = pd.read_excel(excel_path)
    df['URL'] = [extract_hyperlink(excel_path, sheet_name, f"A{i+2}") for i in range(len(df))]
    df['Summary'] = df['URL'].apply(extract_summary_from_url)
    df.to_excel(excel_path, index=False)
    return df

# Update the Excel file and print the DataFrame
excel_path = 'path_to_your_excel_file.xlsx'
updated_df = update_excel_with_summaries(excel_path)
print(updated_df)


pip install openpyxl requests pandas PyPDF2



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# Load your dataset
data = pd.read_csv('your_dataset.csv')

# Assume 'text_column_1' and 'text_column_2' are the names of the columns containing text data
text_columns = ['text_column_1', 'text_column_2']
non_text_columns = ['non_text_column_1', 'non_text_column_2']

# Separate the features (X) from the target variable (y)
X = data.drop(columns=['Disclosure Flag'])
y = data['Disclosure Flag']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Calculate scale_pos_weight
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

# Define the resampling strategy
over_sampler = RandomOverSampler(sampling_strategy=0.5)
under_sampler = RandomUnderSampler(sampling_strategy=1.0)

# Define the TF-IDF vectorization and preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(), text_columns),
        ('non_text', 'passthrough', non_text_columns)
    ]
)

# Create the full pipeline including resampling and XGBoost classifier
pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('resampling', over_sampler),  # or ('resampling', under_sampler) or both
    ('classifier', XGBClassifier(scale_pos_weight=scale_pos_weight))
])

# Train the model on the training data
pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Print the classification report
print(classification_report(y_test, y_pred))





