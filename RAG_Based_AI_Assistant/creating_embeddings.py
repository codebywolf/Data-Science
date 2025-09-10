"""
Script: Generate embeddings for transcript JSON files using Ollama and bge-m3 model

Dependencies:
- Ollama (runs models locally via API): https://ollama.ai/
- bge-m3 (high-quality multilingual embedding model): https://ollama.com/library/bge-m3
- pandas, requests, joblib

Workflow:
1. Load transcript JSON files from the `jsons` directory.
2. For each text chunk, request embeddings from Ollama's API (localhost:11434).
3. Attach embeddings + metadata (chunk_id, text, timestamps).
4. Store results in a Pandas DataFrame and save using joblib for efficient retrieval.
"""

import os
import json
import requests
import pandas as pd
import joblib

# ---------------------------------------------------------
# Function: create_embedding
# Sends a list of text strings to the embedding API (Ollama server in this case)
# using the "bge-m3" model, and returns vector embeddings.
# ---------------------------------------------------------
def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",   # embedding model to use
        "input": text_list   # list of input texts for which embeddings are generated
    })

    embedding = r.json()["embeddings"]  # extract embeddings from API response
    return embedding


# ---------------------------------------------------------
# Step 1: Read all JSON transcripts stored in "jsons" folder
# Each JSON contains segmented text chunks from audio transcripts
# ---------------------------------------------------------
jsons = os.listdir("jsons")  # List all JSON files in directory
my_dicts = []                # will hold processed chunks with embeddings
chunk_id = 0                 # global counter for chunk IDs (unique identifiers)

# ---------------------------------------------------------
# Step 2: Iterate over JSON files and process each transcript
# ---------------------------------------------------------
for json_file in jsons:
    # Load JSON content into memory
    with open(f"jsons/{json_file}") as f:
        content = json.load(f)
    
    print(f"Creating Embeddings for {json_file}")
    
    # Create embeddings for each text chunk in the transcript
    embeddings = create_embedding([c['text'] for c in content['chunks']])
       
    # Attach embeddings + metadata to each chunk
    for i, chunk in enumerate(content['chunks']):
        chunk['chunk_id'] = chunk_id          # assign unique chunk ID
        chunk['embedding'] = embeddings[i]    # assign corresponding embedding
        chunk_id += 1
        my_dicts.append(chunk)                # add to master list

# ---------------------------------------------------------
# Step 3: Convert all processed chunks into a DataFrame
# ---------------------------------------------------------
df = pd.DataFrame.from_records(my_dicts)

# ---------------------------------------------------------
# Step 4: Save the DataFrame with embeddings for later use
# joblib is efficient for storing large objects (better than pickle)
# ---------------------------------------------------------
joblib.dump(df, "embedding.joblib")
