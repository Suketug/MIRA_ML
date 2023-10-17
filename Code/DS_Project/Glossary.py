from docx import Document
import pandas as pd
import re

def extract_terms_from_docx(docx_path):
    # Initialize a Document object for the Word file
    doc = Document(docx_path)
    
    # Initialize lists to store the schema fields
    doc_ids = []
    categories = []
    sub_categories = []
    entities = []
    descriptions = []
    vector_categories = []
    vector_sub_categories = []
    vector_entities = []
    vector_descriptions = []
    source_docs = []
    faq_questions = []
    faq_answers = []
    sample_queries = []
    responses = []
    query_types = []
    applicable_rules = []
    
    doc_id = 1  # Initialize a Doc_ID counter
    
    # Loop through each paragraph in the document
    for para in doc.paragraphs:
        # Split the paragraph text by ':' to separate terms from definitions
        split_text = para.text.split(':', 1)
        
        # Only consider paragraphs that can be split into two parts (term and definition)

        if len(split_text) == 2:
            entity, description = split_text
            # Remove any "*" or "**" or "***" from the entity
            clean_entity = re.sub(r"\*+", "", entity).strip()
            # Append the extracted and default data to the respective lists
            doc_ids.append(doc_id)
            categories.append("Glossary of Terms")
            sub_categories.append("Mortgage Terms")
            entities.append(clean_entity)
            descriptions.append(description.strip())
            vector_categories.append(None)  # Placeholder
            vector_sub_categories.append(None)  # Placeholder
            vector_entities.append(None)  # Placeholder
            vector_descriptions.append(None)  # Placeholder
            source_docs.append(docx_path)
            faq_questions.append(f"What is {clean_entity}?")
            faq_answers.append(description.strip())
            sample_queries.append(f"Define {clean_entity}")
            responses.append(description.strip())
            query_types.append("Definition")
            applicable_rules.append(None)  # Placeholder
            

            doc_id += 1  # Increment the Doc_ID counter
    
    # Create a DataFrame to store the data
    df = pd.DataFrame({
        'Doc_ID': doc_ids,
        'Category': categories,
        'Sub_Category': sub_categories,
        'Entity': entities,
        'Description': descriptions,
        'Vector_Category': vector_categories,
        'Vector_Sub_Category': vector_sub_categories,
        'Vector_Entity': vector_entities,
        'Vector_Description': vector_descriptions,
        'Source_Doc': source_docs,
        'FAQ_Question': faq_questions,
        'FAQ_Answer': faq_answers,
        'Sample_Query': sample_queries,
        'Response': responses,
        'Query_Type': query_types,
        'Applicable_Rules': applicable_rules,
    })
    
    return df

# Extract terms and definitions from the uploaded Word document
docx_path = "Data/Knowledgebase/TermsGlossary.docx"
df_terms = extract_terms_from_docx(docx_path)

df_terms.to_csv("Data/Extracted_Data/extracted_glossary1.csv", index=False)

# Show the first few rows of the DataFrame to verify the extracted data
print(df_terms.head())
