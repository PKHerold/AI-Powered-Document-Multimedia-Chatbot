# AI-Powered Document & Multimedia Q&A

A full-stack web application that lets users upload PDFs, audio, and video files, then interact with an AI chatbot to ask questions, get summaries, and play relevant timestamps.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react)
![MongoDB](https://img.shields.io/badge/MongoDB-7-47A248?logo=mongodb)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)
![Tests](https://img.shields.io/badge/Tests-132_passed-brightgreen)
![Coverage](https://img.shields.io/badge/Coverage-97%25-brightgreen)

---

## Features

- **PDF Upload & Extraction** — Upload PDFs with automatic text extraction and chunking
- **Audio/Video Transcription** — Transcribe media files using Groq Whisper API with timestamped segments
- **AI-Powered Q&A** — Ask questions about documents with context-aware answers from Groq Llama 3
- **Semantic Search** — FAISS vector search for finding the most relevant document sections
- **Timestamp Playback** — Click timestamps in AI responses to jump to the relevant point in audio/video
- **Document Summaries** — Auto-generated summaries for all uploaded content
- **Streaming Responses** — Real-time SSE streaming for chat answers
- **Dark Theme UI** — Premium glassmorphism dark theme with smooth animations

---

## Architecture

```
┌─────────────────┐     ┌──────────────────────────────────────┐
│   React Frontend │────▶│         FastAPI Backend              │
│   (Vite, port    │◀────│         (port 8000)                  │
│    5173/3000)    │     │                                      │
└─────────────────┘     │  ┌──────────┐  ┌──────────────────┐  │
                        │  │ Routers  │  │    Services       │  │
                        │  │ upload   │  │ pdf_service       │  │
                        │  │ chat     │  │ transcription     │  │
                        │  │ documents│  │ llm_service       │  │
                        │  └──────────┘  │ embedding (FAISS) │  │
                        │               │ media_service      │  │
                        │               └──────────────────┘  │
                        └────────┬────────────────┬────────────┘
                                 │                │
                         ┌────────▼──┐    ┌────────▼──────────┐
                         │  MongoDB  │    │  External APIs     │
                         │  (27017)  │    │  • Groq Llama 3    │
                         └───────────┘    │  • Groq Whisper    │
                                          └───────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python + FastAPI |
| Frontend | React 18 (Vite) |
| Database | MongoDB 7 |
| LLM | Groq Llama 3.1 (chat & summarization) |
| Embeddings | sentence-transformers all-MiniLM-L6-v2 (local) |
| Transcription | Groq Whisper large-v3 |
| Vector Search | FAISS |
| Containerization | Docker + Docker Compose |
| CI/CD | GitHub Actions |
| Testing | pytest (97% coverage, 132 tests) |

---

## Project Structure

```
Assignment/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI entry point
│   │   ├── config.py            # Settings & env vars
│   │   ├── models.py            # Pydantic schemas
│   │   ├── database.py          # MongoDB connection (Motor)
│   │   ├── utils.py             # Helper functions
│   │   ├── routers/
│   │   │   ├── upload.py        # POST /api/upload
│   │   │   ├── chat.py          # POST /api/chat, GET /api/chat/stream
│   │   │   └── documents.py     # CRUD + media serving + stats
│   │   └── services/
│   │       ├── pdf_service.py   # PDF text extraction (pdfplumber)
│   │       ├── transcription.py # Audio/video transcription (Groq Whisper)
│   │       ├── llm_service.py   # Groq Llama 3 chat & summarization
│   │       ├── embedding.py     # Local sentence-transformers + FAISS
│   │       └── media_service.py # File storage & serving
│   ├── tests/                   # 132 tests, 97% coverage
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env
├── frontend/
│   ├── src/
│   │   ├── App.jsx              # Main layout
│   │   ├── index.css            # Dark theme design system
│   │   ├── components/
│   │   │   ├── FileUpload.jsx   # Drag & drop upload
│   │   │   ├── ChatBot.jsx      # Chat interface
│   │   │   ├── DocumentList.jsx # Document cards
│   │   │   ├── Summary.jsx      # AI summaries
│   │   │   ├── MediaPlayer.jsx  # Audio/video with timestamps
│   │   │   └── Sidebar.jsx      # Navigation
│   │   └── services/api.js      # API client
│   ├── Dockerfile
│   └── package.json
├── docker-compose.yml
└── .github/workflows/ci.yml
```

---

## Setup

### Prerequisites

- **Python 3.11+**
- **Node.js 18+**
- **MongoDB** (local or Docker)
- **FFmpeg** (for video audio extraction)
- **API Key**: Groq (free at [console.groq.com](https://console.groq.com/keys))

### 1. Clone & Configure

```bash
git clone <repo-url>
cd Assignment
```

Copy and fill in the env template:
```bash
cp backend/.env.example backend/.env
```
```env
GROQ_API_KEY=your_groq_api_key_here
MONGODB_URL=mongodb://localhost:27017
DB_NAME=docqa
UPLOAD_DIR=uploads
MAX_FILE_SIZE_MB=50
CORS_ORIGINS=http://localhost:5173,http://localhost:3000
# Windows: FFMPEG_PATH=C:/ffmpeg/bin/ffmpeg.exe
FFMPEG_PATH=ffmpeg
```

### 2. Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### 3. Frontend

```bash
cd frontend
npm install
npm run dev
```

### 4. MongoDB (Docker)

```bash
docker run -d -p 27017:27017 --name docqa-mongo mongo:7
```

### Docker Compose (All-in-One)

```bash
docker-compose up --build
```

This starts MongoDB (27017), Backend (8000), and Frontend (3000).

---

## API Endpoints

### Upload
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/upload` | Upload PDF, audio, or video file |

### Documents
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/documents` | List all documents |
| `GET` | `/api/documents/{id}` | Get document details |
| `GET` | `/api/documents/{id}/summary` | Get or generate summary |
| `DELETE` | `/api/documents/{id}` | Delete document |
| `GET` | `/api/media/{id}` | Stream media file |
| `GET` | `/api/stats` | App statistics |

### Chat
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/chat` | Ask a question (JSON body: `{document_id, question}`) |
| `GET` | `/api/chat/stream?document_id=...&question=...` | SSE streaming response |
| `GET` | `/api/chat/history/{document_id}` | Get chat history |
| `DELETE` | `/api/chat/history/{document_id}` | Clear chat history |

### Other
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `GET` | `/health` | Health status |
| `GET` | `/docs` | Swagger API docs |

---

## Testing

```bash
cd backend

# Run all tests
py -m pytest tests/ -v

# Run with coverage report
py -m pytest tests/ --cov=app --cov-report=term-missing

# Current: 132 tests, 97% coverage
```

---

## Supported File Types

| Category | Extensions |
|---|---|
| PDF | `.pdf` |
| Audio | `.mp3`, `.wav`, `.ogg`, `.flac`, `.m4a` |
| Video | `.mp4`, `.avi`, `.mkv`, `.mov`, `.webm` |

---

## CI/CD

GitHub Actions pipeline (`.github/workflows/ci.yml`):
- Runs backend tests with coverage check
- Builds frontend
- Builds Docker images

---

## License

MIT
