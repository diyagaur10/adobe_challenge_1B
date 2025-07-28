# France Travel Planner - Challenge 1B

## Approach
This solution processes a set of travel-related PDF documents to extract the most relevant sections and key concepts for a given persona and job-to-be-done. The workflow is fully automated and runs inside a Docker container, making it reproducible and portable.

### Steps:
1. **PDF Title Extraction:**
   - Scans all PDFs in `input/PDFs/` and extracts their titles using font analysis.
   - Generates an input JSON file (`challenge1b_input.json`) listing all documents, persona, and job/task.
2. **Intelligent Analysis:**
   - Loads the input JSON and processes each PDF, chunking text into paragraphs.
   - Uses semantic search (Sentence Transformers) to rank and select the most relevant sections and subsections based on the persona and job.
   - Outputs results to `output/challenge1b_output.json`.

## Models and Libraries Used
- **PyMuPDF (fitz):** For PDF parsing and text extraction.
- **sentence-transformers:** For semantic search and embeddings (model: `all-MiniLM-L6-v2`).
- **scikit-learn:** For TF-IDF and cosine similarity.
- **NumPy:** For numerical operations.
- **Other:** Standard Python libraries (`json`, `os`, `datetime`, etc.).

## How to Build and Run

### Prerequisites
- Docker installed on your system.

### Build the Docker Image
```
docker build -t round1b-solution .
```

### Run the Solution
```
docker run --rm \
  -v "C:/Users/hp/Desktop/pdf1b/input:/app/input" \
  -v "C:/Users/hp/Desktop/pdf1b/output:/app/output" \
  round1b-solution
```
- The container will automatically process all PDFs in `input/PDFs/` and generate the output JSON in `output/`.
- No interactive input is required; persona and job are set in the code (can be changed in `final.py`).

## Expected Execution
- Place all input PDFs in `input/PDFs/`.
- Output will be saved to `output/challenge1b_output.json`.

## Notes
- The solution is fully self-contained and does not require internet access at runtime (except for initial model download during build).
- For custom persona or job, edit the values in `final.py`.

---
For any issues, please refer to the comments in `final.py` or contact the author.
