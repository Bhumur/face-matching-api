# Face Matching API

A Face Verification REST API built using **FastAPI** and **DeepFace (ArcFace)**.

## Features
- Accepts two Base64 images
- Returns confidence score
- Indicates whether faces belong to the same person

## Run Locally
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
