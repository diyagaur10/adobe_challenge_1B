import json
import os
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import datetime

# --- Configuration ---
INPUT_DIR = 'input'
OUTPUT_DIR = 'output'
MODEL_NAME = 'all-MiniLM-L6-v2'

def clean_text(text):
    """Removes extra whitespace and non-printable characters."""
    return " ".join(text.replace('\n', ' ').split())

def get_text_chunks(doc_path):
    """Extracts text from a PDF and splits it into paragraph chunks."""
    chunks = []
    try:
        doc = fitz.open(doc_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            blocks = page.get_text("blocks")
            for i, b in enumerate(blocks):
                text = clean_text(b[4])
                if len(text.split()) > 10:
                    chunks.append({
                        "text": text,
                        "page_number": page_num + 1,
                        "block_num": i
                    })
    except Exception as e:
        print(f"Error processing {os.path.basename(doc_path)}: {e}")
    return chunks

def find_relevant_sections(documents, persona, job_to_be_done, input_pdf_dir):
    """
    Main logic to find and rank relevant sections and subsections.
    """
    print("Loading sentence transformer model...")
    model = SentenceTransformer(MODEL_NAME)

    job_embedding = model.encode(job_to_be_done)

    all_chunks = []
    print("Processing documents and chunking text...")
    for doc_info in documents:
        doc_path = os.path.join(input_pdf_dir, doc_info['filename'])
        if os.path.exists(doc_path):
            chunks = get_text_chunks(doc_path)
            for chunk in chunks:
                chunk['document'] = doc_info['filename']
            all_chunks.extend(chunks)
        else:
            print(f"Warning: Document not found at {doc_path}")

    if not all_chunks:
        return [], []

    print("Performing initial keyword filtering...")
    vectorizer = TfidfVectorizer(stop_words='english', min_df=1)
    tfidf_matrix = vectorizer.fit_transform([chunk['text'] for chunk in all_chunks])
    job_tfidf = vectorizer.transform([job_to_be_done])
    
    keyword_scores = cosine_similarity(job_tfidf, tfidf_matrix).flatten()
    relevant_indices = np.where(keyword_scores > 0.01)[0]
    
    if len(relevant_indices) == 0:
        print("No relevant keywords found. Falling back to all chunks.")
        filtered_chunks = all_chunks
    else:
        filtered_chunks = [all_chunks[i] for i in relevant_indices]
    
    print(f"Filtered down to {len(filtered_chunks)} chunks for semantic analysis.")

    print("Performing semantic search...")
    chunk_texts = [chunk['text'] for chunk in filtered_chunks]
    chunk_embeddings = model.encode(chunk_texts)
    
    semantic_scores = cosine_similarity([job_embedding], chunk_embeddings).flatten()

    for i, chunk in enumerate(filtered_chunks):
        chunk['score'] = semantic_scores[i]

    ranked_chunks = sorted(filtered_chunks, key=lambda x: x['score'], reverse=True)

    extracted_sections = []
    subsection_analysis = []
    seen_documents = set()
    
    for chunk in ranked_chunks:
        if len(extracted_sections) >= 5:
            break
        if chunk['document'] not in seen_documents:
            extracted_sections.append({
                "document": chunk['document'],
                "section_title": f"Relevant content from page {chunk['page_number']}",
                "importance_rank": len(extracted_sections) + 1,
                "page_number": chunk['page_number']
            })
            seen_documents.add(chunk['document'])

    for chunk in ranked_chunks[:5]:
        subsection_analysis.append({
            "document": chunk['document'],
            "refined_text": chunk['text'],
            "page_number": chunk['page_number']
        })
        
    return extracted_sections, subsection_analysis


def process_request(input_dir, output_dir):
    """Main function to run the full process."""
    input_json_path = os.path.join(input_dir, 'challenge1b_input.json')
    input_pdf_dir = os.path.join(input_dir, 'PDFs')
    output_json_path = os.path.join(output_dir, 'challenge1b_output.json')

    if not os.path.exists(input_json_path):
        print(f"Error: Input file not found at {input_json_path}")
        return

    with open(input_json_path, 'r') as f:
        input_data = json.load(f)

    documents = input_data['documents']
    persona = input_data['persona']['role']
    job_to_be_done = input_data['job_to_be_done']['task']

    extracted_sections, subsection_analysis = find_relevant_sections(documents, persona, job_to_be_done, input_pdf_dir)

    # Use timezone-aware datetime object to fix deprecation warning
    processing_time = datetime.datetime.now(datetime.timezone.utc).isoformat()

    output_data = {
        "metadata": {
            "input_documents": [doc['filename'] for doc in documents],
            "persona": persona,
            "job_to_be_done": job_to_be_done,
            "processing_timestamp": processing_time
        },
        "extracted_sections": extracted_sections,
        "subsection_analysis": subsection_analysis
    }

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_json_path, 'w') as f:
        json.dump(output_data, f, indent=4)
        
    print(f"Processing complete. Output saved to {output_json_path}")

if __name__ == '__main__':
    # --- For Docker Execution ---
    process_request(INPUT_DIR, OUTPUT_DIR)
