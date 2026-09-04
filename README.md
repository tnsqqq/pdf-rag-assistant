# RAG-Powered Document Q&A Assistant

A Retrieval-Augmented Generation (RAG) pipeline that lets you ask natural language questions over your own PDF documents. It extracts text from PDFs, embeds them into a vector database, retrieves the most relevant chunks for a given question, and generates a grounded answer using an LLM — minimizing hallucinations by restricting responses to retrieved context.

## How It Works

```
PDF files → pypdf (extract text) → LangChain (chunk) → HuggingFace (embed)
  → Pinecone (store & retrieve) → Groq LLM (generate answer)
```

1. **Ingestion** — PDFs are parsed and their text extracted using `pypdf`.
2. **Chunking** — Extracted text is split into overlapping chunks using LangChain's `RecursiveCharacterTextSplitter`, so context isn't lost at chunk boundaries.
3. **Embedding** — Each chunk is converted into a vector using a HuggingFace sentence-transformer model (`all-MiniLM-L6-v2`).
4. **Storage & Retrieval** — Vectors are stored in Pinecone, a vector database, enabling fast semantic similarity search.
5. **Generation** — When a question is asked, the most relevant chunks are retrieved and passed as context to a Groq-hosted LLM, which generates a final answer using LangChain's retrieval chain.

## Tech Stack

- **Python**
- **LangChain** — orchestration layer connecting all components
- **HuggingFace (Sentence Transformers)** — text embeddings
- **Pinecone** — vector database for semantic search
- **Groq API** — LLM inference (`openai/gpt-oss-120b`)
- **pypdf** — PDF text extraction
- **python-dotenv** — environment variable management

## Features

- Ingests and processes multiple PDF documents at once
- Splits large documents into semantically meaningful chunks
- Stores and searches embeddings efficiently using Pinecone
- Generates context-grounded answers instead of relying on the LLM's raw memory
- Batched vector uploads for handling larger document sets

## Project Structure

```
.
├── document/              # Place your PDF files here
├── main.ipynb              # Notebook containing the full pipeline
├── .env                    # API keys (not committed to git)
└── README.md
```

## Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux
```

### 3. Install dependencies

```bash
pip install python-dotenv pypdf langchain langchain-core langchain-classic langchain-community langchain-text-splitters langchain-huggingface sentence-transformers transformers torch langchain-pinecone pinecone langchain-groq
```

### 4. Set up environment variables

Create a `.env` file in the project root:

```
PINECONE_API_KEY=your_pinecone_api_key
GROQ_API_KEY=your_groq_api_key
```

### 5. Add your PDFs

Place the PDF files you want to query inside the `document/` folder.

### 6. Run the notebook

Run all cells in order (top to bottom). The pipeline will:
- Extract and chunk text from your PDFs
- Generate embeddings and upload them to Pinecone
- Set up the retrieval + generation chain

### 7. Ask questions

```python
print(ask("What is the difference between an asset and a liability?"))
```

## Notes

- Free-tier Groq models are periodically deprecated — check [Groq's model deprecation page](https://console.groq.com/docs/deprecations) if you hit a `model_not_found` error, and update the `model` parameter in the notebook accordingly.
- `pypdf` extracts text from selectable-text PDFs only; scanned/image-based PDFs will require OCR (e.g. Tesseract) before ingestion.

## Future Improvements

- Add a simple web UI (Streamlit/Gradio) for a chat-style interface
- Support additional file formats (DOCX, TXT)
- Add source citation for retrieved chunks in the final answer
- Implement conversational memory for follow-up questions
