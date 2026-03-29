"""
Script: Question-answering system over video transcripts using embeddings + LLM

Dependencies:
- Ollama (local LLM & embedding server): https://ollama.ai/
- bge-m3 (multilingual embedding model): https://huggingface.co/BAAI/bge-m3
- Llama 3.2 (instruction-following LLM for responses): https://ollama.ai/library/llama3.2
- scikit-learn, numpy, joblib

Workflow:
1. Load precomputed transcript embeddings from "embedding.joblib".
2. Accept a user query and generate its embedding.
3. Compute cosine similarity to find the most relevant transcript chunks.
4. Construct a prompt with the top-N chunks and query.
5. Use Ollama’s `llama3.2` model to generate a natural language response.
"""

import joblib
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------
# Function: create_embedding
# Sends text input(s) to Ollama embedding API using bge-m3
# and returns vector embeddings.
# ---------------------------------------------------------
def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",   # embedding model
        "input": text_list   # text(s) to encode
    })

    embedding = r.json()["embeddings"]
    return embedding

# ---------------------------------------------------------
# Step 1: Load existing transcript embeddings
# ---------------------------------------------------------
df = joblib.load('embedding.joblib')

# ---------------------------------------------------------
# Step 2: Accept user query and generate embedding
# ---------------------------------------------------------
incoming_query = input("Ask a question related to video: ")  # user input
question_embedding = create_embedding([incoming_query])[0]  # embedding for query

# ---------------------------------------------------------
# Step 3: Compute similarity between query and transcript chunks
# ---------------------------------------------------------
similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()
top_results = 5  # number of most relevant chunks to retrieve
max_idx = similarities.argsort()[::-1][:top_results]  # indices of top-N chunks

# Extract relevant transcript chunks
new_df = df.loc[max_idx]

# ---------------------------------------------------------
# Step 4: Build prompt for the LLM
# Includes transcript chunks (with metadata) + user query
# ---------------------------------------------------------
prompt = f'''I am teaching web development using Sigma Web Development course. 
Here are the video subtitle chunks containing video titles, video number, start and end time in seconds, and the text at that time:

{new_df[['title', 'number', 'start', 'end', 'text']].to_json()}
--------------------
"{incoming_query}"
User asked this question related to video chunks, you have to answer where and how much content is taught in which video (in which video and which timestamp) 
and guide the user to the particular video. 

If user asks an unrelated question, tell them you can only answer questions related to the course. 
Do not provide exact timestamps — only give a concise 2-line conclusion without extra details.
'''

# ---------------------------------------------------------
# Function: response_model
# Sends constructed prompt to Ollama's LLM (llama3.2)
# and returns generated response.
# ---------------------------------------------------------
def response_model(prompt):
    rm = requests.post("http://localhost:11434/api/generate", json={
        "model": "llama3.2",   # LLM model used for response generation
        "prompt": prompt,      # instruction prompt
        "stream": False        # disable streaming, get full response
    })

    inference = rm.json()
    return inference

# ---------------------------------------------------------
# Step 5: Generate and print final answer
# ---------------------------------------------------------
print(response_model(prompt)['response'])
