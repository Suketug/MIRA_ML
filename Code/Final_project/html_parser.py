from bs4 import BeautifulSoup
import os
import pandas as pd
import json

def process_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    soup = BeautifulSoup(content, 'html.parser')

    # Extract bookmark name from title tag
    title = soup.title.string if soup.title else "N/A"

    # Extract content
    paragraphs = soup.find_all('p')
    content = " ".join(p.text for p in paragraphs)

    # Remove unwanted text
    content = content.replace('< Previous | Contents | Next >', '').strip()
    content = content.replace('< Previous | Contents ', '').strip()

    # Extract tables
    tables = soup.find_all('table')
    table_data = [pd.read_html(str(table))[0].to_dict(orient='records') for table in tables]
    table_data_json = json.dumps(table_data) if table_data else None

    return title, content, table_data_json

# Initialize DataFrame
df = pd.DataFrame(columns=['doc_ids', 'sub_categories', 'entities', 'descriptions', 'text_chunk_id', 'table_data'])

# Initialize counters
doc_id = 1
rows_list = []  # Create an empty list to store rows

# Folder containing all the HTM files
folder_path = 'Data/Full_Extraction/1.1.0_Freddie_Mac_files'  # Replace with your folder path

# Loop through each file in the folder
for file_name in sorted(
    [f for f in os.listdir(folder_path) if f.startswith('part') and f.endswith('.htm')],
    key=lambda x: int(x.split('part')[1].split('.htm')[0])
):
    file_path = os.path.join(folder_path, file_name)
    title, content, table_data_json = process_file(file_path)

    row = {
        'doc_ids': "2",
        'category': 'Freddie Mac_Selling Guide',  # Hard-coded for now
        'sub_categories': title,
        'entities': title,
        'descriptions': content,  # Unchunked content
        'text_chunk_id': doc_id,  # No need for chunk IDs
        'table_data': table_data_json
    }

    # Append the row to the DataFrame
    rows_list.append(row)

# Initialize DataFrame and append rows to DataFrame
df = pd.DataFrame(rows_list)

# Save to CSV
df.to_csv('Data/Full_Extraction/extracted_data2.csv', index=False)
