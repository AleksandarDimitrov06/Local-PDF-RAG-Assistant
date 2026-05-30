# Local RAG pdf assistant on Low-VRAM GPUs 🚀
A Retrieval-Augmented Generation (RAG) application built entirely with open-source tools, designed to run 100% locally on consumer hardware with limited VRAM.

## How It Works
1. **Upload** a PDF file through the Streamlit interface *(recommended: 10 pages or fewer for best performance)*
2. **Chunking** — the PDF is split into smaller overlapping text chunks
3. **Embedding** — each chunk is converted into a vector using `sentence-transformers/all-MiniLM-L6-v2`
4. **Retrieval** — when you ask a question, the most relevant chunks are retrieved from the in-memory vector store
5. **Generation** — the retrieved chunks are passed as context to `Qwen2.5-3B-Instruct`, which generates a precise answer

## Features
- **Privacy First:** No data is sent to external APIs like OpenAI or Groq. Everything runs locally.
- **Low VRAM Optimized:** Uses `bitsandbytes` 4-bit quantization to fit a Large Language Model into less than 4GB of VRAM.
- **Custom Context:** Chat with any PDF document using LangChain and Hugging Face embeddings.

## Hardware Requirements
- NVIDIA GPU with at least 4GB VRAM 
- 8GB+ System RAM
- 5-6 GB free storage space for model weights and cache on first download
> **Note:** Without a GPU it will fall back to CPU — functional but very slow (5–15 min/query)

## Installation
1. **Clone the repository:**
2. **Create a virtual environment (recommended but not essential)**
```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
```
3. **Install dependencies:**
```bash
   pip install -r requirements.txt
```
4. **Run the model**
```bash
   streamlit run app.py
```

> **Note:** Initial startup may take a moment depending on your hardware, as the model loads into memory on first launch.


## Tech Stack
### Frontend / UI
- **Streamlit** — web app interface, chat UI, file uploader, and session state management
### LLM & AI
- **Qwen/Qwen2.5-3B-Instruct** — local causal language model (via HuggingFace)
- **bitsandbytes** — 4-bit quantization (`BitsAndBytesConfig`) to reduce VRAM usage
- **sentence-transformers/all-MiniLM-L6-v2** — embedding model for semantic search
### HuggingFace
- **transformers** — model/tokenizer loading and text-generation pipeline
- **accelerate** — device mapping (`device_map="auto"`) for GPU/CPU inference
### LangChain
- **langchain-huggingface** — HuggingFace pipeline and embeddings integration
- **langchain-community** — `PyPDFLoader` for PDF ingestion
- **langchain-core** — `InMemoryVectorStore`, `RunnablePassthrough`, `RunnableLambda`
- **langchain-text-splitters** — `RecursiveCharacterTextSplitter` for document chunking
### PDF Processing
- **PyPDFLoader** — reads and parses PDF pages into LangChain documents
### Architecture
- **RAG (Retrieval-Augmented Generation)** — in-memory vector store (no persistent DB), local inference (no external API calls)


> **BONUS:** I have added several PDF files inside the `data` folder for testing the application.