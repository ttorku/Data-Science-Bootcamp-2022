
my_list = [
    p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, 
    p21, p22, p23, p24, p25, p26, p27, p28, p29, p30, p31, p32, p33, p34, p35, p36, p37, p38, p39, p40,
    p41, p42, p43, p44, p45, p46, p47, p48, p49, p50, p51, p52, p53, p54, p55, p56, p57, p58, p59, p60,
    p61, p62, p63, p64, p65, p66, p67, p68, p69, p70, p71, p72, p73, p74, p75, p76, p77, p78, p79, p80,
    p81, p82, p83, p84, p85, p86, p87, p88, p89, p90, p91, p92, p93, p94, p95, p96, p97, p98, p99, p100,
    p101, p102, p103, p104, p105, p106, p107, p108, p109, p110, p111, p112, p113, p114, p115, p116, p117, p118, p119, p120,
    p121, p122, p123, p124, p125, p126, p127, p128, p129, p130, p131, p132, p133, p134, p135, p136, p137, p138, p139, p140,
    p141, p142, p143, p144, p145, p146, p147, p148, p149, p150, p151, p152, p153, p154, p155, p156, p157, p158, p159, p160,
    p161, p162, p163, p164, p165, p166, p167, p168, p169, p170, p171, p172, p173, p174, p175, p176, p177, p178, p179, p180,
    p181, p182, p183, p184, p185, p186, p187, p188, p189, p190, p191, p192, p193, p194, p195, p196, p197, p198, p199, p200,
    p201, p202, p203, p204, p205, p206, p207, p208, p209, p210, p211, p212, p213, p214, p215, p216, p217, p218, p219, p220,
    p221, p222, p223, p224, p225, p226, p227, p228, p229, p230, p231, p232, p233, p234, p235, p236, p237, p238, p239, p240,
    p241, p242, p243, p244, p245, p246, p247, p248, p249, p250, p251, p252, p253, p254, p255, p256, p257, p258, p259, p260,
    p261, p262, p263, p264, p265, p266, p267, p268, p269, p270, p271, p272, p273, p274, p275, p276, p277, p278, p279, p280,
    p281, p282, p283, p284, p285, p286, p287, p288, p289, p290, p291, p292, p293, p294, p295, p296, p297, p298, p299, p300,
    p301, p302, p303, p304
]




import os

# Example folder path where PDFs are stored
folder_path = '/path/to/your/pdf/folder'  # Replace with your actual folder path

# List to hold the extracted names
pdf_names = []

# Iterate over the files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.pdf') and filename.startswith('psr_'):
        # Extract the name without extension
        name = filename.rsplit('.', 1)[0]
        pdf_names.append(name)

pdf_names






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






