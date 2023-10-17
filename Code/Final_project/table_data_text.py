import json
import pandas as pd

def convert_table_to_text(table_json):
    try:
        tables = json.loads(table_json)
        sentences = []
        
        for table in tables:
            for row in table:
                key = row.get("0", "")
                value = row.get("1", "")
                if key and value:
                    sentence = f"{key} is {value}."
                    sentences.append(sentence)
                    
        return " ".join(sentences)
        
    except Exception as e:
        return str(e)

# Sample usage
df = pd.read_excel("Data/Full_Extraction/table_data.xlsx")

# Apply the function to the 'table_data' column
df['table_text'] = df['table_data'].apply(convert_table_to_text)

# Save the DataFrame back to Excel
df.to_excel("Data/Full_Extraction/updated_table_data.xlsx", index=False)
