import re
from pdfminer.high_level import extract_text
from logger import setup_logger

# Initialize the logger
logger = setup_logger("main_logger", "app.log")

def read_pdf(pdf_path):
    logger.info("Reading PDF...")
    text = extract_text(pdf_path)
    logger.info("Completed reading PDF.")
    return text

def preprocess_text(text):
    # Replace known incorrect mappings
    text = text.replace("mo\"gage", "mortgage").replace("Mo$gage", "Mortgage")
    
    # Remove page headers and footers (customize the pattern based on your text)
    text = re.sub(r'Glossary of Homeownership and Mortgage Terms \| Better \d+/\d+/\d+, \d+:\d+ AM', '', text)
    text = re.sub(r'https://better.com/glossary\?term=Year-end%20statement Page \d+ of \d+', '', text)
    
    # Remove any standalone "AM"
    text = re.sub(r'\bAM\b', '', text)
    
    # Add a print statement to check
    print("Preprocessed text snippet:", text[:500])
    
    return text

def segment_terms_definitions(text):
    terms_definitions = {}
    
    # Split the text into potential segments based on full stops
    segments = re.split(r'\.\s+', text)
    
    for segment in segments:
        # Find the term and definition by looking for a pattern (term) Definition
        match = re.search(r'([a-zA-Z\s\-]+)\(([a-zA-Z\s\-]+)\)\s+(.+)', segment)
        if match:
            term = match.group(1).strip()
            alias = match.group(2).strip()
            definition = match.group(3).strip()
            
            # Store the term and definition
            terms_definitions[term] = {'alias': alias, 'definition': definition}
    
    # Add a print statement to check
    print("First few terms and definitions:", list(terms_definitions.items())[:10])
    
    return terms_definitions

if __name__ == "__main__":
    pdf_path = 'Data/Knowledgebase/Glossary of Homeownership and Mortgage Terms.pdf' # Update the path accordingly
    pdf_content = read_pdf(pdf_path)
    preprocessed_text = preprocess_text(pdf_content)
    terms_definitions = segment_terms_definitions(preprocessed_text)

    # Output snippets for debugging
    logger.info(f"First few terms and definitions: {list(terms_definitions.items())[:10]}")
