# 🎯 RAG-based AI for Sigma Web Development Course  

This project implements a **Retrieval-Augmented Generation (RAG) pipeline** to answer user questions about the **Sigma Web Development Course**.  
The system works by:  
1. **Extracting audio** from video lectures.  
2. **Transcribing & translating** Hindi speech into English using Whisper.  
3. **Chunking transcripts & generating embeddings** using the **bge-m3** model from Ollama.  
4. **Retrieving the most relevant chunks** based on cosine similarity.  
5. **Answering queries** using **Llama 3.2**, guided by retrieved context.  

---

## 🚀 Features
- 🎥 Converts course videos → audio (MP3).  
- 🗣️ Transcribes Hindi speech and translates to English using Whisper (`large-v2`).  
- 📄 Splits transcripts into **structured JSON** with metadata.  
- 📊 Generates embeddings with **bge-m3** via Ollama.  
- 🔍 Retrieves the most relevant transcript chunks using cosine similarity.  
- 🤖 Uses **Llama 3.2** (via Ollama) to generate **concise answers** with course references.  

---

## 📦 Installation & Setup  

### 1. Clone this repository  
```bash
git clone https://github.com/codebywolf/Data-Science.git
cd Data-Science/RAG_Based_AI_Assistant
```

### 2. Install dependencies  
Install Python dependencies:  
```bash
pip install -r requirements.txt
```
---

## 🛠️ Model Setup  

### 🔹 Whisper (for transcription)  
Install Whisper and FFmpeg:  
```bash
pip install -U openai-whisper
sudo apt update && sudo apt install -y ffmpeg
```

### 🔹 Ollama (for embeddings + LLM)  
Install Ollama: https://ollama.com/download

Pull the required models:  
```bash
ollama pull bge-m3     # Embedding model
ollama pull llama3.2   # LLM for answering
```

📖 Docs:  
- Ollama → https://ollama.ai  
- bge-m3 → https://ollama.com/library/bge-m3  
- Llama 3.2 → https://ollama.ai/library/llama3.2  

---

## ⚡ Workflow  

### 1. Extract audio from videos  
```bash
python video_to_mp3.py
```
- Converts all course videos into MP3 format (`audios/` folder).  

### 2. Transcribe & translate audio  
```bash
python mp3_to_json.py
```
- Uses Whisper (`large-v2`) to transcribe Hindi → English.  
- Saves results as structured JSON (`jsons/` folder).  

### 3. Generate embeddings  
```bash
python creating_embeddings.py
```
- Calls Ollama (`bge-m3`) to embed transcript chunks.  
- Stores results in `embedding.joblib`.  

### 4. Ask questions (RAG pipeline)  
```bash
python inference.py
```
- Takes user input.  
- Retrieves top-5 most relevant chunks via cosine similarity.  
- Passes context + query to Llama 3.2.  
- Prints concise answer.  

---

## 🎯 Example Query  

```bash
$ python inference.py

Ask a question related to video: What is taught in video 12 about HTML tables?
```

✅ The system will return:  
- Which **video number** contains the answer.  
- A **short, 2-line explanation** of the relevant section.  

---

## ⚠️ Notes
- Running Whisper locally requires a GPU. If unavailable, use **Google Colab** or **OpenAI Whisper API**.  

- This RAG system only answers questions **related to the Sigma Web Development Course**.  

--- 
