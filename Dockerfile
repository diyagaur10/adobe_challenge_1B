# Use a specific, lightweight Python base image compatible with linux/amd64
FROM --platform=linux/amd64 python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements first to leverage Docker caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# --- Download and cache the model during the build phase ---
# This is the key step for ensuring offline execution.
# The model is downloaded once here and saved with the image.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy the rest of the application code into the container
COPY . .

# The main command to run the application
# It expects an input.json file in the /app/input directory
CMD ["python", "final.py"]
