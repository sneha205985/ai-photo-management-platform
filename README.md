# 📸 AI Photo Management Platform

A FastAPI-based AI photo management platform that automatically organizes photos using computer vision and natural language processing.

The application indexes images from local folders and Google Photos, detects duplicate and near-duplicate images, performs AI-powered image classification, extracts text from documents using OCR, groups people by facial similarity, and supports natural language semantic search using OpenAI's CLIP model.

---

# Features

## Image Indexing

- Index images from local folders
- Import photos from Google Photos
- Store metadata in SQLite database
- Automatically update existing records

---

## Duplicate Detection

- SHA-256 hashing for exact duplicate detection
- Perceptual Hashing (pHash) for near-duplicate detection
- Adjustable similarity threshold

---

## AI Image Classification

Powered by OpenAI CLIP.

Automatically classifies images into:

- People
- Travel
- Pets
- Documents
- Receipts
- Prescriptions
- Other

---

## Semantic Search

Supports natural language search such as:

- beach
- dog playing
- passport
- smiling person
- receipt
- family vacation

Images are ranked using CLIP embeddings and cosine similarity.

---

## OCR Document Recognition

EasyOCR is used to detect text from images.

Automatically extracts text from:

- Receipts
- Documents
- Prescriptions
- Screenshots

OCR results are stored in the database and combined with semantic search.

---

## Face Detection & Person Grouping

Uses OpenCV Haar Cascade for face detection.

Features:

- Detect faces in images
- Count faces
- Group visually similar faces
- Rename person groups
- Retrieve all photos of the same person

---

## Google Photos Integration

Supports importing images directly from a user's Google Photos library.

Features include:

- OAuth2 authentication
- Download media metadata
- Import selected images
- Process imported photos through the same AI pipeline

---

## REST API

Interactive API documentation is automatically available through Swagger UI.

```
http://localhost:8000/docs
```

---

# Technology Stack

## Backend

- Python 3.11
- FastAPI
- SQLite

## AI & Computer Vision

- OpenAI CLIP
- Transformers
- PyTorch
- OpenCV
- EasyOCR
- Pillow
- ImageHash

## Other

- Docker
- Docker Compose
- Uvicorn

---

# Project Structure

```
photo-ai-platform/
│
├── app/
│   ├── core/
│   ├── services/
│   │   ├── clip_service.py
│   │   ├── face_service.py
│   │   ├── google_photos_service.py
│   │   ├── image_service.py
│   │   ├── indexer.py
│   │   └── ocr_service.py
│   │
│   ├── routers/
│   └── main.py
│
├── credentials/
├── data/
├── tests/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/photo-ai-platform.git

cd photo-ai-platform
```

---

## Create Virtual Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### macOS/Linux

```bash
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run Application

```bash
uvicorn app.main:app --reload
```

Application:

```
http://localhost:8000
```

Swagger:

```
http://localhost:8000/docs
```

---

# Docker

Build:

```bash
docker compose build
```

Run:

```bash
docker compose up
```

Stop:

```bash
docker compose down
```

---

# API Endpoints

## Index Photos

```
POST /index
```

Indexes all images inside the configured folder.

---

## List Photos

```
GET /photos
```

Returns indexed images.

---

## Search Images

```
GET /search?q=beach
```

Natural language semantic search.

---

## Duplicate Detection

```
GET /duplicates
```

Returns:

- Exact duplicates
- Near duplicates

---

## OCR Documents

```
GET /documents
```

Returns all detected documents.

---

## Important Documents

```
GET /documents?important_only=true
```

---

## People

```
GET /people
```

Returns grouped people.

---

## Photos of One Person

```
GET /people/{id}/photos
```

---

## Rebuild Face Groups

```
POST /faces/rebuild
```

---

## Google Photos Login

```
GET /google/login
```

---

## Google Photos Sync

```
POST /google/sync
```

---

# AI Pipeline

```
Image
   │
   ▼
SHA256 Hash
   │
   ▼
Perceptual Hash
   │
   ▼
CLIP Classification
   │
   ▼
CLIP Embedding
   │
   ▼
OCR (Documents Only)
   │
   ▼
Face Detection
   │
   ▼
SQLite Database
   │
   ▼
Semantic Search API
```

---

# Example Search Queries

```
beach

dog

receipt

passport

family

vacation

prescription

travel

person smiling

cat

office document
```

---

# Current Capabilities

- Local image indexing
- Google Photos integration
- Exact duplicate detection
- Near duplicate detection
- AI image classification
- OCR document extraction
- Face detection
- Person grouping
- Semantic search
- REST API
- Docker support

---

# Future Improvements

- Face recognition using FaceNet or ArcFace
- Object detection using YOLOv8
- Image caption generation
- Multi-user authentication
- Cloud database support
- Batch background processing
- Automatic image tagging
- Mobile application

---

# Notes

- The first application startup downloads the CLIP model from Hugging Face.
- OCR accuracy depends on image quality.
- Face grouping uses OpenCV Haar Cascade with appearance-based similarity.
- Google Photos requires OAuth credentials.

---

# Security

The following files are **not committed** to GitHub:

```
credentials/
google_client_secret.json
google_token.json
photo_platform.db
data/photos/
```

These files contain personal data or authentication credentials.

---

# Author

**Sneha Gupta**

---

# License

This project was developed as part of a university assignment for educational purposes.

```
MIT License
```