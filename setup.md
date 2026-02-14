# Smart Revision Generator - Setup Guide

## Project Overview

**Smart Revision Generator** is a GenAI-based intelligent study and revision support system that uses RAG (Retrieval-Augmented Generation), hybrid retrieval (FAISS + BM25), and an agentic workflow to generate structured revision content from academic materials.

---

## Prerequisites

Before setting up the project, ensure you have the following installed:

| Tool | Version | Installation |
|------|---------|-------------|
| Python | 3.11+ | [python.org](https://www.python.org/downloads/) or `brew install python@3.11` |
| Git | Latest | [git-scm.com](https://git-scm.com/) |
| pip | Latest | Comes with Python |

### External Services Required

1. **Google Gemini API Key** - [Get API Key](https://aistudio.google.com/app/apikey)
2. **Supabase Account** - [Create Account](https://supabase.com/)

---

## Project Structure

```
smart-revision-generator/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application entry point
│   ├── config.py               # Centralized configuration
│   ├── routers/                # API route handlers
│   │   └── __init__.py
│   ├── services/               # Business logic services
│   │   ├── __init__.py
│   │   ├── ingestion.py        # Document upload & text extraction
│   │   ├── chunking.py         # Text chunking logic
│   │   ├── embedding.py        # SBERT embedding generation
│   │   ├── retrieval.py        # Hybrid retrieval (FAISS + BM25)
│   │   ├── generation.py       # LLM-based content generation
│   │   └── evaluation.py       # Output quality evaluation
│   ├── utils/                  # Utility functions
│   │   └── __init__.py
│   ├── prompts/                # LLM prompt templates
│   │   └── __init__.py
│   ├── vector_store/           # FAISS index storage
│   │   └── __init__.py
│   ├── agents/                 # Agentic workflow components
│   │   └── __init__.py
│   └── models/                 # Pydantic schemas
│       └── __init__.py
├── tests/                      # Test suite
│   └── __init__.py
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variables template
├── .gitignore                  # Git ignore rules
├── setup.md                    # This file
└── LICENSE
```

---

## Installation Steps

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd GenAI-Based-System-for-Intelligent-Study-and-Revision-Support
```

### Step 2: Create Virtual Environment

**macOS/Linux:**
```bash
# Using Python 3.11 specifically (recommended)
python3.11 -m venv venv

# Or using default Python 3 (must be 3.11+)
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate
```

**Windows:**
```bash
# Create virtual environment
python -m venv venv

# Activate (Command Prompt)
venv\Scripts\activate.bat

# Activate (PowerShell)
venv\Scripts\Activate.ps1
```

### Step 3: Upgrade pip

```bash
pip install --upgrade pip
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The first installation may take several minutes due to PyTorch and sentence-transformers downloading model weights.

### Step 5: Configure Environment Variables

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your credentials
nano .env  # or use any text editor
```

**Required Environment Variables:**

| Variable | Description | Example |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Your Google Gemini API key | `AIza...` |
| `SUPABASE_URL` | Your Supabase project URL | `https://xxx.supabase.co` |
| `SUPABASE_KEY` | Your Supabase anon/public key | `eyJ...` |

---

## Supabase Database Setup

### Step 1: Create a New Supabase Project

1. Go to [Supabase Dashboard](https://app.supabase.com/)
2. Click **New Project**
3. Fill in project details and create

### Step 2: Run Database Migrations

Execute the following SQL in Supabase SQL Editor:

```sql
-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Documents table
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    filename VARCHAR(255) NOT NULL,
    upload_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    file_size INTEGER,
    file_type VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Chunks table
CREATE TABLE chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    chunk_text TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    token_count INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Generations table
CREATE TABLE generations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    output_json JSONB NOT NULL,
    compression_ratio FLOAT,
    accuracy_score FLOAT,
    generation_time_seconds FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Feedback table
CREATE TABLE feedback (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    generation_id UUID REFERENCES generations(id) ON DELETE CASCADE,
    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
    comment TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX idx_chunks_document_id ON chunks(document_id);
CREATE INDEX idx_generations_document_id ON generations(document_id);
CREATE INDEX idx_feedback_generation_id ON feedback(generation_id);
CREATE INDEX idx_documents_upload_timestamp ON documents(upload_timestamp DESC);
```

### Step 3: Get API Credentials

1. Go to **Project Settings** > **API**
2. Copy the **Project URL** → `SUPABASE_URL`
3. Copy the **anon/public key** → `SUPABASE_KEY`

---

## Running the Application

### Start the FastAPI Backend

```bash
# Development mode with auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at: http://localhost:8000

API Documentation:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Start the Streamlit Frontend

```bash
# In a separate terminal
streamlit run streamlit_app.py --server.port 8501
```

The frontend will be available at: http://localhost:8501

---

## Tech Stack Reference

| Component | Technology | Purpose |
|-----------|------------|---------|
| Backend Framework | FastAPI | REST API server |
| Database | Supabase (PostgreSQL) | Metadata & output storage |
| Vector Store | FAISS (HNSW) | Embedding storage & retrieval |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | Text vectorization |
| Sparse Retrieval | rank-bm25 | Keyword-based retrieval |
| LLM | Google Gemini 1.5 Flash | Content generation |
| Frontend | Streamlit | User interface |
| Evaluation | sklearn, rouge-score | Quality metrics |

---

## Key Dependencies

```
fastapi==0.109.2          # Web framework
supabase==2.4.0           # Database client
faiss-cpu==1.7.4          # Vector search
sentence-transformers==2.3.1  # Embeddings
rank-bm25==0.2.2          # BM25 retrieval
google-generativeai==0.4.1  # LLM API (Gemini)
PyPDF2==3.0.1             # PDF processing
python-docx==1.1.0        # DOCX processing
streamlit==1.31.1         # Frontend
```

---

## Development Workflow

### Code Formatting

```bash
# Format code with Black
black app/ tests/

# Sort imports with isort
isort app/ tests/

# Type checking with mypy
mypy app/
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_chunking.py -v
```

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      USER UPLOAD                                 │
│                    (PDF / DOCX)                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   TEXT EXTRACTION                                │
│              (PyPDF2 / python-docx)                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      CHUNKING                                    │
│            (300-500 tokens, 50 overlap)                          │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│   EMBEDDING (SBERT)     │     │    METADATA STORE       │
│   → FAISS Index         │     │    → Supabase           │
└─────────────────────────┘     └─────────────────────────┘
              │                               │
              └───────────────┬───────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   HYBRID RETRIEVAL                               │
│         Dense (FAISS) × 0.6 + Sparse (BM25) × 0.4               │
│                     → Top 5 Chunks                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   LLM GENERATION                                 │
│             (Gemini 1.5 Flash + RAG Context)                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 STRUCTURED OUTPUT (JSON)                         │
│  concepts | definitions | applications | flashcards | faqs      │
│  mock_questions (2/5/10 mark) | summary                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EVALUATION                                    │
│         Cosine Similarity | ROUGE Score | Compression           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Agentic Workflow

The system uses 4 specialized agents orchestrated by an AgentController:

| Agent | Responsibility |
|-------|----------------|
| `ContentAnalyzerAgent` | Analyzes document structure and content type |
| `ConceptExtractorAgent` | Extracts key concepts, definitions, applications |
| `QuestionGeneratorAgent` | Generates FAQs, flashcards, mock questions |
| `EvaluationAgent` | Evaluates output quality and relevance |

---

## Troubleshooting

### Common Issues

**1. FAISS Installation Error (macOS)**
```bash
# If faiss-cpu fails, try:
pip install faiss-cpu --no-cache-dir

# Or for M1/M2 Macs:
pip install faiss-cpu==1.7.4
```

**2. PyTorch Installation Issues**
```bash
# Install CPU-only version
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**3. Supabase Connection Error**
- Verify your `SUPABASE_URL` and `SUPABASE_KEY` in `.env`
- Check if the project is active in Supabase dashboard
- Ensure tables are created correctly

**4. Gemini API Rate Limits**
- The system uses Gemini 1.5 Flash which has generous rate limits
- Implement exponential backoff (already included in `tenacity`)

**5. Memory Issues with Large PDFs**
- Processing is done in chunks to manage memory
- For very large documents (>100 pages), consider splitting

---

## Contact & Support

For questions or issues:
1. Check existing GitHub issues
2. Create a new issue with detailed description
3. Include error logs and environment details

---

## License

See [LICENSE](LICENSE) file for details.
