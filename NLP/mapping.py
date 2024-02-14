# Load the new project data from the uploaded file to ensure we are using the latest data
new_excel_path = '/mnt/data/project_data.xlsx'
new_project_data_df = pd.read_excel(new_excel_path)

# Get the unique sub_departments from the new data
new_sub_depts = new_project_data_df['sub_dept'].unique()
new_num_sub_depts = len(new_sub_depts)

# Define the mapping for L1 using the new data, setting non-applicable entries to '0.0'
new_mapping_L1 = pd.DataFrame({
    'Main_dept': ['Construction'] * len(new_project_data_df),
    'Mapped': new_project_data_df['applicability'].apply(lambda x: 1.0 if x == 'Yes' else 0.0),
    'Project_id': new_project_data_df['project_id']
})

# Define the mapping for L2 using the new data
new_mapping_L2 = pd.DataFrame({
    'Sub_dept': new_sub_depts,
    'Mapped': [1.0] * new_num_sub_depts,
    'Project_id': [[] for _ in range(new_num_sub_depts)]
})

# Populate the Project_id lists for each sub_department mapping using the new data
for i, sub_dept in enumerate(new_sub_depts, 1):
    new_mapping_L2.at[new_mapping_L2.index[new_mapping_L2['Sub_dept'] == sub_dept], 'Project_id'] = \
        new_project_data_df['project_id'][new_project_data_df[f'sub_dept'] == sub_dept].tolist()

# Store the new mappings in a dictionary callable by keys 'L1' and 'L2'
new_mapping = {
    'L1': new_mapping_L1,
    'L2': new_mapping_L2
}

# Display the new mapping for L1 to confirm the changes
new_mapping['L1']



# Create the structure as seen in the image with binary indicators
# Assuming that the number of sub_departments is equal to the number of unique values in the 'sub_dept' column
sub_depts = project_data_df['sub_dept'].unique()
num_sub_depts = len(sub_depts)

# Map sub_department to a binary format for each project
binary_data = {
    'Project_id': project_data_df['project_id'],
    'Project_name': project_data_df['project_name'],
    'L1_main_dept': [1.0] * len(project_data_df)  # All projects belong to the Construction main department
}

# Initialize sub_department columns with 0.0
for i in range(1, num_sub_depts + 1):
    binary_data[f'L2_sub_dept{i}'] = [0.0] * len(project_data_df)

# Set the appropriate sub_department to 1.0 based on the project's sub_dept
for index, row in project_data_df.iterrows():
    sub_dept_index = sub_depts.tolist().index(row['sub_dept']) + 1
    binary_data[f'L2_sub_dept{sub_dept_index}'][index] = 1.0

# Create the new DataFrame
binary_df = pd.DataFrame(binary_data)

# Show the first few rows of the new dataframe to verify its structure
binary_df.head()

