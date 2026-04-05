# GitLab Handbook GenAI Chatbot 🦊

An interactive RAG-based chatbot that helps users explore GitLab's open Handbook and Direction pages. Built with LangChain, FAISS, HuggingFace embeddings, and Google's Gemini LLM.

## Live Demo
> **[Launch Chatbot on Streamlit Cloud](https://app-chatbot-fw7jza5ozr5drdaw5p7vhg.streamlit.app/)**  
> *(Enter your own Gemini API key in the sidebar to start chatting)*

## Features
- **Deep Sub-Link Discovery** — Automatically crawls 15 seed pages and discovers 150+ sub-section hyperlinks from the GitLab Handbook and Direction pages, ensuring comprehensive coverage.
- **Local Embeddings (HuggingFace)** — Uses `all-MiniLM-L6-v2` for embedding, running entirely locally with zero API costs.
- **Gemini-Powered Answers** — Uses Google's `gemini-2.5-flash` LLM for generating context-aware responses.
- **Conversational Memory** — Remembers chat history for seamless follow-up questions.
- **Topic Guardrails** — Politely declines off-topic questions and stays focused on GitLab content.
- **Graceful Error Handling** — Auto-retries on rate limits; shows friendly messages instead of stack traces.
- **GitLab-Themed UI** — Custom CSS with GitLab's brand colors and layout for a polished experience.

## Architecture

```
User Query → Streamlit UI → HuggingFace Embeddings → FAISS Retrieval → Gemini LLM → Response
```

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Frontend | Streamlit | Chat interface with GitLab theming |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` | Local vector embeddings (no API needed) |
| Vector Store | FAISS | Fast similarity search over 821 chunks |
| LLM | Google Gemini `2.5-flash` | Answer generation |
| Data Pipeline | BeautifulSoup + LangChain | Web scraping with auto sub-link discovery |

## Project Structure

```
├── app.py              # Streamlit frontend with chat UI
├── rag_chain.py        # LangChain RAG pipeline (retriever + LLM)
├── data_loader.py      # Data ingestion: scraping, chunking, embedding
├── faiss_index/        # Pre-built FAISS vector store (821 chunks)
├── requirements.txt    # Python dependencies
├── .streamlit/         # Streamlit Cloud configuration
│   └── config.toml
├── Project_Writeup.md  # Detailed technical writeup
└── README.md           # This file
```

## Setup & Run Locally

### Prerequisites
- Python 3.9+
- A Google Gemini API key ([Get one free](https://aistudio.google.com))

### Installation
```bash
git clone <your-repo-url>
cd gitlab-chatbot

python -m venv venv
# Windows
.\venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### Run the Chatbot
```bash
streamlit run app.py
```
Open http://localhost:8501 and paste your Gemini API key in the sidebar.

### (Optional) Rebuild the Knowledge Base
The FAISS index is pre-built and included in the repo. To rebuild it with fresh data:
```bash
python data_loader.py
```

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Select your repo, set main file to `app.py`
4. In **Advanced Settings → Secrets**, add:
   ```toml
   GOOGLE_API_KEY = "your-gemini-api-key"
   ```
5. Click Deploy

## Key Technical Decisions

| Decision | Rationale |
|----------|-----------|
| HuggingFace over Gemini embeddings | Zero API cost, no rate limits, works offline |
| Auto sub-link discovery | Covers 165 pages vs. only 15 hardcoded URLs |
| FAISS over cloud vector DB | No external dependencies, fast, included in repo |
| Gemini 2.5 Flash | Best balance of quality and free-tier quota (1,500 req/day) |
| k=3 retrieval | Reduces token usage by 40% while maintaining answer quality |
| Topic guardrails | Keeps assistant focused on GitLab content only |
