# NLP-HW2
"""
# RAG Document Q&A System

A minimal Retrieval-Augmented Generation (RAG) system that answers questions based on uploaded documents.

## Features

- 📁 Upload PDF and TXT documents
- 🔍 Automatic text chunking and embedding
- 💾 Vector storage with FAISS
- ❓ Question answering using Qwen 3 30B via Open Router
- 📊 Display of retrieved chunks used for answers

## Setup

1. Install requirements:
```bash
pip install -r requirements.txt
