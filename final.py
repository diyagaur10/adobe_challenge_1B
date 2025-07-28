import json
import os
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import datetime
import re
from collections import Counter

# --- Configuration ---
# Define paths for local execution
INPUT_PDF_DIR = 'input/PDFs'
INPUT_JSON_PATH = 'input/challenge1b_input.json'
OUTPUT_JSON_PATH = 'output/challenge1b_output.json'
MODEL_NAME = 'all-MiniLM-L6-v2'

# ==============================================================================
#  MODULE 1A LOGIC: PDF Outline Extraction
# ==============================================================================

def clean_text_for_outline(text):
    """Cleans text for heading extraction."""
    text = re.sub(r'\s+', ' ', text)
    return ''.join(filter(lambda x: x.isprintable(), text)).strip()

def extract_pdf_title(pdf_path):
    """
    Extracts only the title from a given PDF file using font analysis.
    This is a streamlined version of the 1A logic for this specific task.
    """
    try:
        doc = fitz.open(pdf_path)
        if not doc.page_count:
            return os.path.basename(pdf_path) # Fallback to filename

        font_counts = Counter()
        for page in doc:
            blocks = page.get_text("dict")["blocks"]
            for b in blocks:
                if "lines" in b:
                    for l in b["lines"]:
                        for s in l["spans"]:
                            font_size = round(s["size"])
                            font_counts[font_size] += 1
        
        if not font_counts:
            # Fallback for image-based PDFs
            return doc.load_page(0).get_text().split('\n')[0].strip() or os.path.basename(pdf_path)

        # Find the largest font size, likely the title
        title_size = sorted(font_counts.keys(), reverse=True)[0]

        # Search for the title text on the first page
        page = doc.load_page(0)
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if "lines" in b:
                for l in b["lines"]:
                    if l["spans"] and round(l["spans"][0]["size"]) == title_size:
                        title_text = " ".join(clean_text_for_outline(s["text"]) for s in l["spans"])
                        if len(title_text) > 5: # Heuristic to avoid short non-titles
                            doc.close()
                            return title_text
        
        # If no title found with font size, use first line as fallback
        doc.close()
        return doc.load_page(0).get_text().split('\n')[0].strip() or os.path.basename(pdf_path)

    except Exception as e:
        print(f"Error extracting title from {os.path.basename(pdf_path)}: {e}")
        return os.path.basename(pdf_path)


def generate_1b_input_json(pdf_dir, persona, job_to_be_done):
    """
    Generates the 'challenge1b_input.json' file by scanning PDFs for titles
    and combining them with user input for persona and job.
    """
    print("--- Generating 1B Input File ---")
    documents = []
    if not os.path.exists(pdf_dir):
        print(f"Error: PDF directory not found at '{pdf_dir}'")
        return None

    for filename in os.listdir(pdf_dir):
        if filename.lower().endswith('.pdf'):
            print(f"Extracting title from: {filename}")
            pdf_path = os.path.join(pdf_dir, filename)
            title = extract_pdf_title(pdf_path)
            documents.append({
                "filename": filename,
                "title": title
            })

    input_data = {
        "challenge_info": {
            "challenge_id": "round_1b_combined_001",
            "test_case_name": "custom_analysis",
            "description": f"Analysis for {persona}"
        },
        "documents": documents,
        "persona": {"role": persona},
        "job_to_be_done": {"task": job_to_be_done}
    }

    # Ensure the 'input' directory exists
    os.makedirs(os.path.dirname(INPUT_JSON_PATH), exist_ok=True)
    
    with open(INPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(input_data, f, indent=4)
    
    print(f"Successfully generated '{INPUT_JSON_PATH}'")
    return input_data

# ==============================================================================
#  MODULE 1B LOGIC: Intelligent Analysis
# ==============================================================================

def clean_text_for_analysis(text):
    """Removes extra whitespace for semantic analysis."""
    return " ".join(text.replace('\n', ' ').split())

def get_text_chunks(doc_path):
    """Extracts text from a PDF and splits it into paragraph chunks."""
    chunks = []
    try:
        doc = fitz.open(doc_path)
        for page_num, page in enumerate(doc):
            blocks = page.get_text("blocks")
            for i, b in enumerate(blocks):
                text = clean_text_for_analysis(b[4])
                if len(text.split()) > 10: # Filter out very short text blocks
                    chunks.append({
                        "text": text,
                        "page_number": page_num + 1,
                        "block_num": i
                    })
    except Exception as e:
        print(f"Error processing {os.path.basename(doc_path)}: {e}")
    return chunks

def find_relevant_sections(documents, persona, job_to_be_done, pdf_dir):
    """
    Main logic to find and rank relevant sections and subsections.
    """
    print("\n--- Starting Intelligent Analysis (1B) ---")
    print("Loading sentence transformer model... (This may take a moment on first run)")
    model = SentenceTransformer(MODEL_NAME)

    # Combine persona and job for a richer query
    query = f"As a {persona}, I need to {job_to_be_done}."
    query_embedding = model.encode(query)

    all_chunks = []
    print("Processing documents and chunking text...")
    for doc_info in documents:
        doc_path = os.path.join(pdf_dir, doc_info['filename'])
        if os.path.exists(doc_path):
            chunks = get_text_chunks(doc_path)
            for chunk in chunks:
                chunk['document'] = doc_info['filename']
            all_chunks.extend(chunks)

    if not all_chunks:
        print("No text could be extracted from the documents.")
        return [], []

    print("Performing semantic search on all text chunks...")
    chunk_texts = [chunk['text'] for chunk in all_chunks]
    chunk_embeddings = model.encode(chunk_texts, show_progress_bar=True)
    
    semantic_scores = cosine_similarity([query_embedding], chunk_embeddings).flatten()

    for i, chunk in enumerate(all_chunks):
        chunk['score'] = semantic_scores[i]

    ranked_chunks = sorted(all_chunks, key=lambda x: x['score'], reverse=True)

    # Format output
    extracted_sections = []
    subsection_analysis = []
    seen_sections = set()
    
    # Extract top 5 unique sections (approximated by document + page)
    for chunk in ranked_chunks:
        if len(extracted_sections) >= 5:
            break
        section_key = (chunk['document'], chunk['page_number'])
        if section_key not in seen_sections:
            extracted_sections.append({
                "document": chunk['document'],
                "section_title": f"Relevant content from page {chunk['page_number']}",
                "importance_rank": len(extracted_sections) + 1,
                "page_number": chunk['page_number']
            })
            seen_sections.add(section_key)

    # Extract top 5 most relevant subsections (paragraphs)
    for chunk in ranked_chunks[:5]:
        subsection_analysis.append({
            "document": chunk['document'],
            "refined_text": chunk['text'],
            "page_number": chunk['page_number']
        })
        
    return extracted_sections, subsection_analysis

# ==============================================================================
#  MAIN EXECUTION WORKFLOW
# ==============================================================================

def main():
    """Main function to run the full, combined workflow."""
    
    print("--- Intelligent PDF Processing System ---")
    
    # 1. Set default persona and job for non-interactive execution
    persona_input = "Student"  # Default persona
    job_input = "find key concepts"  # Default job
    print(f"Using persona: {persona_input}")
    print(f"Using job to be done: {job_input}")

    # 2. Generate the 1B input JSON file dynamically
    input_data = generate_1b_input_json(INPUT_PDF_DIR, persona_input, job_input)
    
    if not input_data:
        print("Failed to generate input data. Exiting.")
        return

    # 3. Run the 1B intelligent analysis using the generated data
    documents = input_data['documents']
    persona = input_data['persona']['role']
    job_to_be_done = input_data['job_to_be_done']['task']

    extracted_sections, subsection_analysis = find_relevant_sections(documents, persona, job_to_be_done, INPUT_PDF_DIR)

    # 4. Create the final output file
    # Use timezone-aware datetime object
    processing_time = datetime.datetime.now(datetime.timezone.utc).isoformat()

    final_output = {
        "metadata": {
            "input_documents": [doc['filename'] for doc in documents],
            "persona": persona,
            "job_to_be_done": job_to_be_done,
            "processing_timestamp": processing_time
        },
        "extracted_sections": extracted_sections,
        "subsection_analysis": subsection_analysis
    }

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)

    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4)
        
    print(f"\nProcessing complete. Final output saved to '{OUTPUT_JSON_PATH}'")

if __name__ == '__main__':
    main()
