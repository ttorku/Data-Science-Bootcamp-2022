# List with content
content_list = ["1. summary\n This is my legal frame.\n Governments want to stop illegality."]

# Function to process each string in the list
def process_content(content):
    # Check if the string starts with "1. summary\n" and remove it
    if content.startswith("1. summary\n"):
        content = content.replace("1. summary\n", "")
    # Replace other '\n' with space
    return content.replace("\n", " ")

# Apply the function to each element in the list
processed_list = [process_content(content) for content in content_list]

processed_list






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





import requests
from PyPDF2 import PdfReader
import pandas as pd
import os

# Function to download PDF from a given URL
def download_pdf(url, filename):
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(filename, 'wb') as f:
            f.write(response.content)
        return True
    except requests.RequestException as e:
        return False

# Function to extract the abstract from a PDF file
def extract_abstract_from_pdf(file_path):
    try:
        reader = PdfReader(file_path)
        full_text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
        start_idx = full_text.lower().find("abstract")
        if start_idx != -1:
            abstract_text = full_text[start_idx:]
            words = abstract_text.split()
            abstract = " ".join(words[:300])  # Limiting to 300 words
            return abstract
        else:
            return "Abstract not found"
    except Exception as e:
        return f"Error in extracting abstract: {e}"

# Load CSV data
csv_file_path = 'path_to_your_csv_file.csv'  # Replace with your CSV file path
df = pd.read_csv(csv_file_path)

# Directory for downloading PDFs
pdf_dir = 'path_to_pdf_directory'  # Replace with your desired directory path
os.makedirs(pdf_dir, exist_ok=True)

# Processing each paper
abstracts = []
for index, row in df.iterrows():
    paper_id = row['Paper ID']
    url = row['Paper Name (URL links)']
    pdf_filename = f"{pdf_dir}/{paper_id}.pdf"

    # Download PDF
    download_success = download_pdf(url, pdf_filename)
    if download_success:
        # Extract abstract
        abstract = extract_abstract_from_pdf(pdf_filename)
    else:
        abstract = "Failed to download PDF"

    abstracts.append(abstract)

# Add abstracts to the DataFrame
df['Abstract'] = abstracts

# Save updated DataFrame to CSV
df.to_csv('path_to_updated_csv_file.csv', index=False)  # Replace with your desired output CSV file path






